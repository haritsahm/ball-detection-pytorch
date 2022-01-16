from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from src.models.modules.metrics import RadiusAccuracy, IoUPercentile, PrecisionRecal
from src.models.modules.postprocessing import fcnnv2_post_processs
from src.models.modules.bitbots_fcnn import FCNNv2


class FCNNv2LitModel(LightningModule):
    """Lightning model for Bitbots's FCNN v1
    """

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        input_size: List[int] = [150, 200],
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = FCNNv2(hparams=self.hparams)

        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("Model is not nn.Module")

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_radacc = [RadiusAccuracy(n) for n in [3, 5, 10]]
        self.train_iou = IoUPercentile()
        self.val_radacc = [RadiusAccuracy(n) for n in [3, 5, 10]]
        self.val_iou = IoUPercentile()
        self.val_precall = PrecisionRecal()
        self.test_radacc = [RadiusAccuracy(n) for n in [3, 5, 10]]
        self.test_iou = IoUPercentile()
        self.test_precall = PrecisionRecal()

        # for logging best so far validation accuracy
        self.val_acc_best = [MaxMetric() for _ in range(3)]

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        images, masks, bboxes, gtcenter_x, gtcenter_y, imgInfo = batch
        logits = self.forward(images)

        logits = logits.squeeze(dim=1)
        loss = self.criterion(logits, masks.to(dtype=torch.float32))

        preds = torch.gt(logits, 0.5).to(dtype=torch.int64)

        return loss, preds, masks, bboxes, gtcenter_x, gtcenter_y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, masks, bboxes, gtcenter_x, gtcenter_y = self.step(batch)

        output = preds.squeeze().detach().cpu().numpy()
        candidates = fcnnv2_post_processs(output)
        candidates = torch.from_numpy(candidates)  # (N, 3)

        # for func in self.train_radacc:
        #     [func.update(cand, cand, ctx, cty)
        #      for cand, ctx, cty in zip(candidates, gtcenter_x, gtcenter_y)]

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass
        # acc = [func.compute() for func in self.train_radacc]  # get val accuracy from current epoch
        # self.log("train/acc_3", acc[0], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc_5", acc[1], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc_10", acc[2], on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, masks, bboxes, gtcenter_x, gtcenter_y = self.step(batch)

        # TODO: Compute metrics
        # - TP/FP/FN, intersection > 50% with highest candidate
        # - Radius Accuracy for highest candidate

        output = preds.squeeze().detach().cpu().numpy()
        candidates = fcnnv2_post_processs(output)
        candidates = torch.from_numpy(candidates)  # (N, 3)

        # TP, FP, FN
        # IoU w/ percentile 90/99
        for pred, mask in zip(preds, masks):
            self.val_iou.update(pred, mask)

        for cand, box in zip(candidates, bboxes):
            if torch.cuda.is_available():
                cand = cand.cuda()
                box = box.cuda()
            self.val_precall.update(cand.unsqueeze(0), box)
        # [func.update(candidates[:, 0], candidates[:, 1], gtcenter_x, gtcenter_y)
        #  for func in self.val_radacc]

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "target_x": gtcenter_x, "target_y": gtcenter_y}

    def validation_epoch_end(self, outputs: List[Any]):
        # acc = [func.compute() for func in self.val_radacc]  # get val accuracy from current epoch
        # [func(val) for func, val in zip(self.val_acc_best, acc)]
        iou, iou_90, iou_99 = self.val_iou.compute()
        # self.log("val/acc_3", acc[0], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc_5", acc[1], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc_10", acc[2], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc_rad3_best", self.val_acc_best[0].compute(), on_epoch=True, prog_bar=True)
        # self.log("val/acc_rad5_best", self.val_acc_best[1].compute(), on_epoch=True, prog_bar=True)
        # self.log("val/acc_rad10_best",
        #          self.val_acc_best[2].compute(), on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou_90", iou_90, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou_99", iou_99, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, masks, bboxes, gtcenter_x, gtcenter_y = self.step(batch)

        arg_pred_x = torch.argmax(preds[0], dim=1)  # (N, 1)
        arg_pred_y = torch.argmax(preds[1], dim=1)  # (N, 1)

        # acc = [func(arg_pred_x, arg_pred_y, gtcenter_x, gtcenter_y) for func in self.test_radacc]

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        # self.log("test/acc_3", acc[0], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc_5", acc[1], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc_10", acc[2], on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "target_x": gtcenter_x, "target_y": gtcenter_y}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        [func.reset() for func in self.train_radacc]
        [func.reset() for func in self.val_radacc]
        [func.reset() for func in self.test_radacc]
        self.val_iou.reset()
        self.val_precall.reset()
        self.test_iou.reset()
        self.test_precall.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        def lmbda(epoch): return 0.9975 ** epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

        return [optimizer], [scheduler]
