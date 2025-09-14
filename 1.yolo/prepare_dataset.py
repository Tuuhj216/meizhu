#!/usr/bin/env python3
"""
Dataset preparation script for crosswalk detection.
This script helps organize and prepare your dataset for YOLO training.
"""

import os
import shutil
from pathlib import Path
import random

def create_dataset_structure():
    """Create the required directory structure for YOLO dataset."""
    dirs = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("Dataset directory structure created!")

def split_dataset(images_dir: str, train_ratio: float = 0.8):
    """
    Split images and labels into train/val sets.

    Args:
        images_dir: Path to directory containing images and labels
        train_ratio: Ratio of data to use for training
    """
    if not os.path.exists(images_dir):
        print(f"Images directory {images_dir} not found!")
        return

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in os.listdir(images_dir)
              if Path(f).suffix.lower() in image_extensions]

    if not images:
        print("No images found in the specified directory!")
        return

    # Shuffle and split
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copy files to appropriate directories
    for img_list, subset in [(train_images, 'train'), (val_images, 'val')]:
        for img_name in img_list:
            # Copy image
            src_img = os.path.join(images_dir, img_name)
            dst_img = f"dataset/images/{subset}/{img_name}"
            shutil.copy2(src_img, dst_img)

            # Copy corresponding label file if it exists
            label_name = Path(img_name).stem + '.txt'
            src_label = os.path.join(images_dir, label_name)
            if os.path.exists(src_label):
                dst_label = f"dataset/labels/{subset}/{label_name}"
                shutil.copy2(src_label, dst_label)

    print(f"Dataset split complete!")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

def create_sample_annotations():
    """Create sample annotation files to show the format."""
    sample_dir = "sample_annotations"
    Path(sample_dir).mkdir(exist_ok=True)

    # Create sample annotation
    with open(f"{sample_dir}/sample.txt", "w") as f:
        f.write("# YOLO annotation format (one line per object):\n")
        f.write("# class_id center_x center_y width height (all normalized 0-1)\n")
        f.write("0 0.5 0.6 0.8 0.1  # crosswalk at center-bottom\n")
        f.write("1 0.3 0.65 0.1 0.02  # crosswalk line\n")
        f.write("1 0.5 0.65 0.1 0.02  # crosswalk line\n")
        f.write("1 0.7 0.65 0.1 0.02  # crosswalk line\n")

    print(f"Sample annotation created in {sample_dir}/sample.txt")

if __name__ == "__main__":
    print("Crosswalk Dataset Preparation Tool")
    print("=" * 40)

    # Create directory structure
    create_dataset_structure()

    # Create sample annotations
    create_sample_annotations()

    print("\nNext steps:")
    print("1. Place your images and label files in a single directory")
    print("2. Run: python prepare_dataset.py --split /path/to/your/images")
    print("3. Or manually organize files into dataset/images/{train,val}")
    print("4. Ensure labels are in dataset/labels/{train,val}")