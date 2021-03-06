from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from src.models.modules.metrics import RadiusAccuracy
from src.models.modules.bitbots_fcnn import FCNNv1


class FCNNv1LitModel(LightningModule):
    """Lightning model for Bitbots's FCNN v1
    """

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = FCNNv1(hparams=self.hparams)

        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("Model is not nn.Module")

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_radacc = [RadiusAccuracy(n) for n in [3, 5, 10]]
        self.val_radacc = [RadiusAccuracy(n) for n in [3, 5, 10]]
        self.test_radacc = [RadiusAccuracy(n) for n in [3, 5, 10]]

        # for logging best so far validation accuracy
        self.val_acc_best = [MaxMetric() for _ in range(3)]

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        image, _, _, gtcenter_x, gtcenter_y, imgInfo = batch
        logits_x, logits_y = self.forward(image)
        gtcenter_x = torch.stack(gtcenter_x, dim=0)
        gtcenter_y = torch.stack(gtcenter_y, dim=0)
        arg_gt_x = torch.argmax(gtcenter_x, dim=1)  # (N, 1)
        arg_gt_y = torch.argmax(gtcenter_y, dim=1)  # (N, 1)

        loss = self.criterion(logits_x, arg_gt_x) + self.criterion(logits_y, arg_gt_y)
        preds = F.softmax(logits_x, dim=1), F.softmax(logits_y, dim=1)

        return loss, preds, arg_gt_x, arg_gt_y, imgInfo

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, gtcenter_x, gtcenter_y, imgInfo = self.step(batch)

        arg_pred_x = torch.argmax(preds[0], dim=1)  # (N, 1)
        arg_pred_y = torch.argmax(preds[1], dim=1)  # (N, 1)

        print(arg_pred_x)
        print(gtcenter_x)

        # log train metrics
        acc = [func(arg_pred_x, arg_pred_y, gtcenter_x, gtcenter_y) for func in self.train_radacc]

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc_3", acc[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_5", acc[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_10", acc[2], on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "target_x": gtcenter_x, "target_y": gtcenter_y}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, gtcenter_x, gtcenter_y, imgInfo = self.step(batch)

        arg_pred_x = torch.argmax(preds[0], dim=1)  # (N, 1)
        arg_pred_y = torch.argmax(preds[1], dim=1)  # (N, 1)

        acc = [func(arg_pred_x, arg_pred_y, gtcenter_x, gtcenter_y) for func in self.val_radacc]

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc_3", acc[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_5", acc[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_10", acc[2], on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "target_x": gtcenter_x, "target_y": gtcenter_y}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = [func.compute() for func in self.val_radacc]  # get val accuracy from current epoch
        [func(val) for func, val in zip(self.val_acc_best, acc)]
        self.log("val/acc_rad3_best", self.val_acc_best[0].compute(), on_epoch=True, prog_bar=True)
        self.log("val/acc_rad5_best", self.val_acc_best[1].compute(), on_epoch=True, prog_bar=True)
        self.log("val/acc_rad10_best",
                 self.val_acc_best[2].compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, gtcenter_x, gtcenter_y, imgInfo = self.step(batch)

        arg_pred_x = torch.argmax(preds[0], dim=1)  # (N, 1)
        arg_pred_y = torch.argmax(preds[1], dim=1)  # (N, 1)

        acc = [func(arg_pred_x, arg_pred_y, gtcenter_x, gtcenter_y) for func in self.test_radacc]

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc_3", acc[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_5", acc[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_10", acc[2], on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "target_x": gtcenter_x, "target_y": gtcenter_y}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        [func.reset() for func in self.train_radacc]
        [func.reset() for func in self.val_radacc]
        [func.reset() for func in self.test_radacc]

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
