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
        idx += 1 # coco annotatoin start from 1

        imgInfo = self._coco_anns.loadImgs(ids=idx)[0]
        annInfos = self._coco_anns.loadAnns(ids=self._coco_anns.getAnnIds(imgIds=idx))
        img_path = os.path.join(self._img_dir, imgInfo["file_name"])
        
        # Load image
        image = cv2.imread(filename=img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert bgr to rgb

        im_w, im_h, _ = image.shape

        # Create mask
        mask = np.zeros(image.shape[:2], dtype="uint8")
        
        # Load annotations
        bboxes = []
        for ann in annInfos:
            # bboxes
            if ann['category_id'] == 1: # ball only
                x, y, w, h = ann['bbox'] # x, y, w ,h 
                bboxes.append([x,y,w,h,"ball"])

                # semantic mask
                cx, cy = int(x+w/2), int(y+h/2)
                mask = cv2.ellipse(mask, (cx, cy), (int(w/2), int(h/2)), 0, 0, 360, 255, -1)

        # Apply augmentations
        if self._transform:
            transformed = self._transform(image=image, bboxes=bboxes, mask=mask)

            image = transformed['image']
            bboxes = transformed['bboxes']
            mask = transformed['mask']
            
        tim_h, tim_w, _ = image.shape
        # Get bbox centers
        center_x = torch.zeros(tim_w) # [tim_w]
        center_y = torch.zeros(tim_h) # [tim_h]
        if len(bboxes) == 1:
            for box in bboxes:
                cx, cy = int(box[0] + box[2]/2), int(box[1] + box[3]/2)
                center_x[cx] = 1
                center_y[cy] = 1

        # To tensors
        bboxes = torch.as_tensor(bboxes) # [N, 4]

        return image, mask, bboxes, center_x, center_y, imgInfo
