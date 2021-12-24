import os
import albumentations as A
import json
import pytest
import tempfile
import torch

from albumentations.pytorch import ToTensorV2
from src.datamodules.datasets.fcnn_dataset import FCNNDataset
from torch.utils.data import Dataset
from torchvision import transforms, utils

img_dir = "/media/haritsahm/DataStorage/TORSO_21_dataset/example_images"


def annotation_sample():
    return {
        "info": {
            "version": "1.0",
        },
        "images": [
            {
                "file_name": "img_fake_cam_000059.PNG",
                "height": 480,
                "width": 640,
                "id": 1
            }],
        "categories": [
            {"id": 0, "name": "obstacle", "supercategory": None},
            {"id": 1, "name": "robot", "supercategory": None},
            {"id": 2, "name": "ball", "supercategory": None}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [
                0.0, 2.0, 369.00000000000006, 478.0], "area": 176382.00000000003, "blurred": False, "concealed": False},
            {"id": 2, "image_id": 3, "category_id": 0, "bbox": [
                159.0, 0.0, 227.0, 478.0], "area": 108506.0, "blurred": False, "concealed": False},
            {"id": 3, "image_id": 3, "category_id": 0, "bbox": [
                385.0, 0.0, 225.0, 334.0], "area": 75150.0, "blurred": False, "concealed": False}
        ]
    }


class TestFCNNDataset(object):

    def test_fcnn_dataset(self):
        annotation = annotation_sample()
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as tmp:
            json.dump(annotation, tmp)

        dataset = FCNNDataset(annotations_file=path, img_dir=img_dir, transform=None)

        # Object type
        assert isinstance(dataset, torch.utils.data.Dataset)

        # Dataset len
        assert len(dataset) == 1

        # Dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        image, mask, bbox, center_x, center_y, imginfo = next(iter(dataloader))

        assert image.shape[0] == 1
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(bbox, torch.Tensor)
        assert isinstance(center_x, torch.Tensor)
        assert isinstance(center_y, torch.Tensor)
        assert isinstance(imginfo, dict)

        os.remove(path)

    def test_augmentation(self):
        annotation = annotation_sample()

        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as tmp:
            json.dump(annotation, tmp)

        transform = A.Compose([ToTensorV2()], bbox_params=A.BboxParams(format='coco'))

        dataset = FCNNDataset(annotations_file=path, img_dir=img_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        image, mask, bbox, center_x, center_y, imginfo = next(iter(dataloader))

        assert image.shape[0] == 1
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(bbox, torch.Tensor)

        os.remove(path)

        with pytest.raises(ValueError):
            transform = transforms.Compose(
                [
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            dataset = FCNNDataset(annotations_file=path, img_dir=img_dir, transform=transform)

    def test_zero_bounding_box(self):
        annotation = annotation_sample()
        annotation['annotations'] = []

        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as tmp:
            # do stuff with temp file
            json.dump(annotation, tmp)

        dataset = FCNNDataset(annotations_file=path, img_dir=img_dir, transform=None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        _, _, bbox, center_x, center_y, _ = next(iter(dataloader))

        os.remove(path)
