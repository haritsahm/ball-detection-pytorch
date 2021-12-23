import pytest

from src.models.modules.metrics import RadiusAccuracy
from torch import tensor

_predx_reg = tensor([
    [5, 3, 7, 4, 0]]
)

_targetx_reg = tensor([
    [5, 3, 8, 3, 8]]
)

_predy_reg = tensor([
    [4, 2, 4, 1, 3]]
)

_targety_reg = tensor([
    [1, 6, 4, 5, 4]]
)

@pytest.mark.parametrize(
    "pred_x, pred_y, target_x, target_y, radius, exp_result",
    [
        (_predx_reg, _predy_reg, _targetx_reg, _targety_reg, 3, 0.2),
        (_predx_reg, _predy_reg, _targetx_reg, _targety_reg, 5, 0.8),
        (_predx_reg, _predy_reg, _targetx_reg, _targety_reg, 10, 1.0),
    ],
)
def test_radius_accuracy(pred_x, pred_y, target_x, target_y, radius, exp_result):
    acc = RadiusAccuracy(radius=radius)

    acc(pred_x, pred_y, target_x, target_y)

    assert acc.compute() == tensor(exp_result)