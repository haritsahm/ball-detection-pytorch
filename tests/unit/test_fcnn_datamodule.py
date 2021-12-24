import os

import pytest
import torch

from src.datamodules.fcnn_datamodule import FCNNDataModule


def test_fcnn_datamodule(batch_size):
    datamodule = FCNNDataModule(batch_size=batch_size)

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
    image, mask, bbox, centers, imginfo = batch

    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
