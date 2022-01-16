from typing import Union, Tuple, Optional

import os
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import albumentations as A


class FCNNDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transform: Optional[A.Compose] = None):
        if not isinstance(annotations_file, str) and not isinstance(img_dir, str):
            raise ValueError("Annotation file and image directory should be a str")

        if transform and not isinstance(transform, A.Compose):
            raise ValueError("Transform must be an albumentations Compose")

        self._coco_anns = COCO(annotations_file)
        self._img_dir = img_dir
        self._transform = transform
        self._class_labels = ['obstacle', 'robot', 'ball']

    def __len__(self) -> int:
        return len(self._coco_anns.getImgIds())

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        idx += 1  # coco annotatoin start from 1

        imgInfo = self._coco_anns.loadImgs(ids=idx)[0]
        annInfos = self._coco_anns.loadAnns(ids=self._coco_anns.getAnnIds(imgIds=idx))
        img_path = os.path.join(self._img_dir, imgInfo["file_name"])

        # Load image
        image = cv2.imread(filename=img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        im_w, im_h, _ = image.shape

        # Create mask
        mask = np.zeros(image.shape[:2], dtype="uint8")

        # Load annotations
        bboxes = []
        for ann in annInfos:
            # bboxes
            if ann['category_id'] == self._class_labels.index("ball"):
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, w, h, "ball"])

                # semantic mask
                cx, cy = int(x + w / 2), int(y + h / 2)
                mask = cv2.ellipse(mask, (cx, cy), (int(w / 2), int(h / 2)), 0, 0, 360, 1, -1)

        # Apply augmentations
        if self._transform:
            transformed = self._transform(image=image, bboxes=bboxes, mask=mask)

            image = transformed['image']
            bboxes = [list(box)[:4] for box in transformed['bboxes']]
            mask = transformed['mask']

        # Get bbox centers
        _, tim_h, tim_w = image.shape
        center_x = torch.zeros(tim_w)
        center_y = torch.zeros(tim_h)

        for box in bboxes:
            cx, cy = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)
            center_x[cx] = 1
            center_y[cy] = 1

        # To tensors
        image = torch.as_tensor(image, dtype=torch.float32)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.uint8)

        return image, mask, bboxes, center_x, center_y, imgInfo

    def collate_fn(self, batch):
        """
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        mask = list()
        bboxes = list()
        center_x = list()
        center_y = list()
        imgInfo = list()

        for b in batch:
            images.append(b[0])
            mask.append(b[1])
            bboxes.append(b[2])
            center_x.append(b[3])
            center_y.append(b[4])
            imgInfo.append(b[5])

        images = torch.stack(images, dim=0)
        mask = torch.stack(mask, dim=0)
        center_x = torch.stack(center_x, dim=0)
        center_y = torch.stack(center_y, dim=0)

        return images, mask, bboxes, center_x, center_y, imgInfo
