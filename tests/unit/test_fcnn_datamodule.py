import os

import pytest
import torch

from src.datamodules.fcnn_datamodule import FCNNDataModule


@pytest.mark.parametrize("batch_size", [16, 32, 64])
@pytest.mark.parametrize("data_dir", ["/media/haritsahm/DataStorage/TORSO_21_dataset/torso_21_detection_dataset"])
def test_fcnn_datamodule(batch_size, data_dir):
    datamodule = FCNNDataModule(batch_size=batch_size, data_dir=data_dir)

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test

    datamodule.setup()

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test
    assert (
        len(datamodule.data_train) + len(datamodule.data_val) + len(datamodule.data_test) == 10464
    )

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    image, mask, bbox, center_x, center_y, imginfo = batch

    assert len(image) == batch_size
    assert len(mask) == batch_size
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(bbox, list)
    assert isinstance(center_x, list)
    assert isinstance(center_y, list)
