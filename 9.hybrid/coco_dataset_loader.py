import json
import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO


class COCODatasetLoader:
    """
    COCO dataset loader and converter for crosswalk detection.
    Supports loading COCO format annotations and converting them to YOLO format.
    """

    def __init__(self, annotation_path: str, image_dir: str, target_class: str = "crosswalk"):
        """
        Initialize COCO dataset loader.

        Args:
            annotation_path: Path to COCO annotation JSON file
            image_dir: Path to directory containing images
            target_class: Target class name to filter (default: "crosswalk")
        """
        self.annotation_path = annotation_path
        self.image_dir = Path(image_dir)
        self.target_class = target_class
        self.coco = None
        self.target_cat_ids = []

        if os.path.exists(annotation_path):
            self.load_annotations()

    def load_annotations(self):
        """Load COCO annotations and find target category IDs."""
        try:
            self.coco = COCO(self.annotation_path)

            # Find category IDs that match our target class
            categories = self.coco.loadCats(self.coco.getCatIds())
            self.target_cat_ids = []

            for cat in categories:
                if self.target_class.lower() in cat['name'].lower():
                    self.target_cat_ids.append(cat['id'])
                    print(f"Found target category: {cat['name']} (ID: {cat['id']})")

            if not self.target_cat_ids:
                print(f"Warning: No categories found matching '{self.target_class}'")
                print("Available categories:")
                for cat in categories:
                    print(f"  - {cat['name']} (ID: {cat['id']})")

        except Exception as e:
            print(f"Error loading COCO annotations: {e}")

    def get_image_annotations(self, image_id: int) -> List[Dict]:
        """Get all annotations for a specific image."""
        if not self.coco:
            return []

        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.target_cat_ids)
        return self.coco.loadAnns(ann_ids)

    def get_all_target_images(self) -> List[Dict]:
        """Get all images that contain target annotations."""
        if not self.coco:
            return []

        img_ids = self.coco.getImgIds(catIds=self.target_cat_ids)
        return self.coco.loadImgs(img_ids)

    def coco_to_yolo_polygon(self, annotation: Dict, image_width: int, image_height: int) -> Optional[List[float]]:
        """
        Convert COCO annotation to YOLO polygon format.

        Args:
            annotation: COCO annotation dictionary
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            List of normalized polygon coordinates [x1, y1, x2, y2, ...] or None
        """
        try:
            # Handle segmentation data
            if 'segmentation' in annotation and annotation['segmentation']:
                segmentation = annotation['segmentation']

                # If it's RLE format, convert to polygon
                if isinstance(segmentation, dict):  # RLE format
                    mask = mask_utils.decode(segmentation)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Use the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        # Simplify contour to reduce points
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
                        polygon = simplified.flatten()
                    else:
                        return None

                elif isinstance(segmentation, list):  # Polygon format
                    if len(segmentation) > 0:
                        polygon = segmentation[0]  # Use first polygon
                    else:
                        return None
                else:
                    return None

                # Normalize coordinates
                normalized_polygon = []
                for i in range(0, len(polygon), 2):
                    if i + 1 < len(polygon):
                        x = float(polygon[i]) / image_width
                        y = float(polygon[i + 1]) / image_height
                        # Clamp to [0, 1]
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        normalized_polygon.extend([x, y])

                return normalized_polygon if len(normalized_polygon) >= 6 else None

            # Fallback to bounding box if no segmentation
            elif 'bbox' in annotation:
                bbox = annotation['bbox']  # [x, y, width, height]
                x, y, w, h = bbox

                # Convert bbox to polygon (rectangle)
                x1, y1 = x / image_width, y / image_height
                x2, y2 = (x + w) / image_width, (y + h) / image_height

                # Clamp to [0, 1]
                x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
                x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))

                # Return as rectangle polygon
                return [x1, y1, x2, y1, x2, y2, x1, y2]

        except Exception as e:
            print(f"Error converting annotation to YOLO format: {e}")

        return None

    def convert_to_yolo_format(self, output_dir: str, train_split: float = 0.8):
        """
        Convert COCO dataset to YOLO format and save to specified directory.

        Args:
            output_dir: Output directory for YOLO format dataset
            train_split: Ratio of images for training (rest goes to validation)
        """
        if not self.coco:
            print("No COCO annotations loaded")
            return

        output_path = Path(output_dir)

        # Create directory structure
        train_images_dir = output_path / "images" / "train"
        val_images_dir = output_path / "images" / "val"
        train_labels_dir = output_path / "labels" / "train"
        val_labels_dir = output_path / "labels" / "val"

        for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Get all images with target annotations
        images = self.get_all_target_images()
        print(f"Found {len(images)} images with {self.target_class} annotations")

        # Split into train/val
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(images)
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        print(f"Train: {len(train_images)} images, Val: {len(val_images)} images")

        # Process training images
        self._process_image_split(train_images, train_images_dir, train_labels_dir, "train")

        # Process validation images
        self._process_image_split(val_images, val_images_dir, val_labels_dir, "val")

        # Create dataset configuration file
        self._create_dataset_config(output_path)

        print(f"Dataset conversion completed. Output saved to: {output_path}")

    def _process_image_split(self, images: List[Dict], images_dir: Path, labels_dir: Path, split_name: str):
        """Process a split of images and save in YOLO format."""
        processed = 0

        for img_info in images:
            try:
                # Copy image file
                src_image_path = self.image_dir / img_info['file_name']
                if not src_image_path.exists():
                    print(f"Warning: Image not found: {src_image_path}")
                    continue

                dst_image_path = images_dir / img_info['file_name']
                if not dst_image_path.exists():  # Avoid overwriting
                    import shutil
                    shutil.copy2(src_image_path, dst_image_path)

                # Process annotations
                annotations = self.get_image_annotations(img_info['id'])

                # Create label file
                label_file = labels_dir / (Path(img_info['file_name']).stem + '.txt')

                with open(label_file, 'w') as f:
                    for ann in annotations:
                        polygon = self.coco_to_yolo_polygon(
                            ann, img_info['width'], img_info['height']
                        )

                        if polygon and len(polygon) >= 6:
                            # Class ID 0 for crosswalk (single class)
                            line = "0 " + " ".join(f"{coord:.6f}" for coord in polygon)
                            f.write(line + "\n")

                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed}/{len(images)} {split_name} images")

            except Exception as e:
                print(f"Error processing image {img_info['file_name']}: {e}")

    def _create_dataset_config(self, output_path: Path):
        """Create YOLO dataset configuration file."""
        config = {
            'train': str(output_path / "images" / "train"),
            'val': str(output_path / "images" / "val"),
            'nc': 1,  # Number of classes
            'names': ['crosswalk']
        }

        config_file = output_path / "dataset.yaml"

        with open(config_file, 'w') as f:
            for key, value in config.items():
                if isinstance(value, str):
                    f.write(f"{key}: '{value}'\n")
                elif isinstance(value, list):
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")

        print(f"Dataset configuration saved to: {config_file}")

    def load_coco_image_with_annotations(self, image_id: int) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Load an image and its annotations from COCO dataset.

        Args:
            image_id: COCO image ID

        Returns:
            Tuple of (image_array, annotations_list)
        """
        if not self.coco:
            return None, []

        # Get image info
        img_info = self.coco.loadImgs([image_id])[0]
        image_path = self.image_dir / img_info['file_name']

        # Load image
        image = None
        if image_path.exists():
            image = cv2.imread(str(image_path))

        # Get annotations
        annotations = self.get_image_annotations(image_id)

        return image, annotations

    def visualize_annotations(self, image_id: int, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Visualize COCO annotations on an image.

        Args:
            image_id: COCO image ID
            save_path: Optional path to save the visualized image

        Returns:
            Annotated image array or None
        """
        image, annotations = self.load_coco_image_with_annotations(image_id)

        if image is None:
            print(f"Could not load image for ID: {image_id}")
            return None

        annotated_image = image.copy()

        for ann in annotations:
            try:
                if 'segmentation' in ann and ann['segmentation']:
                    segmentation = ann['segmentation']

                    if isinstance(segmentation, dict):  # RLE format
                        mask = mask_utils.decode(segmentation)
                        # Create colored overlay
                        color_mask = np.zeros_like(annotated_image)
                        color_mask[mask == 1] = [0, 255, 0]  # Green
                        annotated_image = cv2.addWeighted(annotated_image, 0.7, color_mask, 0.3, 0)

                    elif isinstance(segmentation, list) and len(segmentation) > 0:
                        # Polygon format
                        polygon = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(annotated_image, [polygon], (0, 255, 0), cv2.LINE_AA)
                        cv2.polylines(annotated_image, [polygon], True, (0, 0, 255), 2)

                # Draw bounding box if available
                if 'bbox' in ann:
                    x, y, w, h = [int(v) for v in ann['bbox']]
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            except Exception as e:
                print(f"Error visualizing annotation: {e}")

        if save_path:
            cv2.imwrite(save_path, annotated_image)
            print(f"Annotated image saved to: {save_path}")

        return annotated_image


def main():
    """Example usage of COCODatasetLoader."""
    import argparse

    parser = argparse.ArgumentParser(description="COCO Dataset Loader and Converter")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations JSON file")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Output directory for YOLO format")
    parser.add_argument("--class", default="crosswalk", help="Target class name to filter")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--visualize", type=int, help="Visualize annotations for specific image ID")

    args = parser.parse_args()

    # Initialize loader
    loader = COCODatasetLoader(args.annotations, args.images, getattr(args, 'class'))

    if args.visualize:
        # Visualize specific image
        result = loader.visualize_annotations(args.visualize, f"visualization_{args.visualize}.jpg")
        if result is not None:
            print(f"Visualization saved for image ID: {args.visualize}")
    else:
        # Convert dataset
        loader.convert_to_yolo_format(args.output, args.train_split)


if __name__ == "__main__":
    main()