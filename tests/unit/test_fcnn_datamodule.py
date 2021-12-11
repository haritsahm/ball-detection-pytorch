import os

import pytest
import torch

from src.datamodules.datasets.fcnn_dataset import FCNNDataset

img_dir = "/media/haritsahm/DataStorage/TORSO_21_dataset/torso_21_detection_dataset/data"
annotation_file = "/media/haritsahm/DataStorage/TORSO_21_dataset/torso_21_detection_dataset/train.json"


def fcnn_dataset():
    dataset = FCNNDataset(annotations_file=annotations_file, img_dir=img_dir, transform=None)

    assert dataset == True
