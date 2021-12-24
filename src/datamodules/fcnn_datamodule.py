from typing import Optional, Tuple

import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets import fcnn_dataset


class FCNNDataModule(LightningDataModule):
    """
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        train_transform: str = None,
        test_transform: str = None,
        image_size: list = [150, 200],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = A.load(train_transform, data_format='yaml') if train_transform else \
            A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomResizedCrop(width=image_size[1], height=image_size[0]),
                A.HorizontalFlip(p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='coco'))

        self.test_transforms = A.load(test_transform, data_format='yaml') if test_transform else \
            A.Compose([
                A.SmallestMaxSize(max_size=image_size[1] + 60),
                A.CenterCrop(width=image_size[1], height=image_size[0]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='coco'))

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        img_dir = os.path.join(self.hparams.data_dir, "data")

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = fcnn_dataset.FCNNDataset(
                annotations_file=os.path.join(self.hparams.data_dir, "train.json"), img_dir=img_dir, transform=self.train_transforms)
            self.data_val = fcnn_dataset.FCNNDataset(
                annotations_file=os.path.join(self.hparams.data_dir, "val.json"), img_dir=img_dir, transform=self.test_transforms)
            self.data_test = fcnn_dataset.FCNNDataset(
                annotations_file=os.path.join(self.hparams.data_dir, "test.json"), img_dir=img_dir, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.data_train.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_val.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_test.collate_fn
        )
