import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import numpy as np


class CrosswalkDataset(Dataset):
    def __init__(self, data_dir, annotation_file, transform=None, img_size=640):
        self.data_dir = data_dir
        self.coco = COCO(annotation_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.img_size = img_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.data_dir, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert('RGB')
        original_size = image.size

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Process bounding boxes
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Scale boxes to match resized image
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            # Scale boxes to new image size
            scale_x = self.img_size / original_size[0]
            scale_y = self.img_size / original_size[1]
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.long)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(img_id)
        }

        return image, target


def get_dataloader(data_dir, annotation_file, batch_size=8, shuffle=True, num_workers=4):
    dataset = CrosswalkDataset(data_dir, annotation_file)

    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        images = torch.stack(images, 0)
        return images, targets

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return dataloader