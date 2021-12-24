import numpy as np
import torch
from torchmetrics import Metric

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

        assert pred_x.shape == target_x.shape and pred_y.shape == target_y.shape
        dist = torch.pow(pred_x-target_x, 2) + torch.pow(pred_y-target_y, 2)
        dist_np = dist.cpu().detach().numpy()
        self.dist += torch.from_numpy(self.vec_func_r(dist_np)).sum()
        self.total += target_x.numel()

    def compute(self):
        return self.dist.float() / self.total