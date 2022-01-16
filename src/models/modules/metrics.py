import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import Metric
from torchmetrics.functional import iou


class RadiusAccuracy(Metric):
    def __init__(self, radius: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.rad = radius
        self.add_state("dist", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.vec_func_r = np.vectorize(self.func_r)

    def func_r(self, dist: torch.Tensor):
        if dist >= self.rad**2:
            return 0
        else:
            return 1

    def update(self, pred_x: torch.Tensor, pred_y: torch.Tensor, target_x: torch.Tensor, target_y: torch.Tensor):

        dist = torch.pow(pred_x - target_x, 2) + torch.pow(pred_y - target_y, 2)
        dist_np = dist.cpu().detach().numpy()
        self.dist += torch.from_numpy(self.vec_func_r(dist_np)).sum()
        self.total += target_x.numel()

    def compute(self):
        return self.dist.float() / self.total


class IoUPercentile(Metric):
    def __init__(self, num_classes=2, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self._num_classes = num_classes
        self.add_state("iou", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        preds = preds.reshape(-1)
        target = target.reshape(-1)
        target = target.to(dtype=preds.dtype)

        self.iou.append(iou(preds, target).cpu())

    def compute(self):

        if len(self.iou) == 0:
            return 0, 0, 0

        avg_ = np.sum(self.iou) / len(self.iou)
        prct_90_idx_ = 0.9 * len(self.iou)
        if prct_90_idx_.is_integer():
            prct_90_val_ = (self.iou[int(prct_90_idx_)] + self.iou[int(prct_90_idx_) - 1]) / 2.0
        else:
            prct_90_val_ = self.iou[int(np.round(prct_90_idx_))]
        prct_99_idx_ = 0.99 * len(self.iou)
        if prct_99_idx_.is_integer():
            prct_99_val_ = (self.iou[int(prct_99_idx_)] + self.iou[int(prct_99_idx_) - 1]) / 2.0
        else:
            prct_99_val_ = self.iou[int(np.round(prct_99_idx_))]

        return avg_, prct_90_val_, prct_99_val_


class PrecisionRecal(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

        self.mul_pred = torch.FloatTensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-1, -1, 1, 1]])
        self.mul_targ = torch.FloatTensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]])

        if torch.cuda.is_available():
            self.mul_pred = self.mul_pred.cuda()
            self.mul_targ = self.mul_targ.cuda()

    # Source: https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/losses.py#L5
    def calc_iou(self, a, b):
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - \
            torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - \
            torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)

        ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

        ua = torch.clamp(ua, min=1e-8)

        intersection = iw * ih

        IoU = intersection / ua

        return IoU

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # pred [N, 3] cx,cy,r
        # target [M, 4] x,y,w,h

        preds = preds.to(dtype=target.dtype)

        if len(preds) == 0:
            self.fn += 1
        elif len(target) == 0 and len(preds > 0):
            self.fp += 1
        else:
            xyxy_pred = torch.mm(preds, self.mul_pred)
            xyxy_targ = torch.mm(target, self.mul_targ)

            if xyxy_pred.shape[0] != xyxy_targ.shape[0]:
                xyxy_pred = xyxy_pred.repeat(xyxy_targ.shape[0], 1)

            iou_val = self.calc_iou(xyxy_pred, xyxy_targ)  # (N,)

            if any(iou_val > 0.5):
                self.tp += 1
            else:
                self.fp += 1

    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)

        return precision, recall
