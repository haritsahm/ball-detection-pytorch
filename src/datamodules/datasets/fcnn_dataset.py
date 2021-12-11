# TODO:
# 1. Create dataloader to give output in (x,y) ball center and (mask) ball area
# 2. Use Albumentations augmentation pipeline

import os
from pycocotools.coco import COCO


class FCNNDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transform=None):
        if not isinstance(annotations_file, str):
            raise ValueError

        if not isinstance(img_dir, str):
            raise ValueError

        self.coco_labels = COCO(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
