import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List, Optional
import math
from pathlib import Path
from coco_dataset_loader import COCODatasetLoader


class CrosswalkDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', coco_dataset_path: Optional[str] = None):
        """
        Initialize crosswalk detector with YOLO model

        Args:
            model_path: Path to YOLO model weights
            coco_dataset_path: Optional path to COCO dataset annotations for training data augmentation
        """
        self.model = YOLO(model_path)
        self.coco_loader = None

        if coco_dataset_path and Path(coco_dataset_path).exists():
            image_dir = Path(coco_dataset_path).parent / "images"
            self.coco_loader = COCODatasetLoader(coco_dataset_path, str(image_dir))
            print(f"Loaded COCO dataset: {coco_dataset_path}")

        # You may need to train a custom model for crosswalk detection
        # or use a pre-trained model that includes pedestrian crossing detection

    def detect_crosswalk(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[dict]:
        """
        Detect crosswalks in the image

        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for detection

        Returns:
            List of detected crosswalk bounding boxes with confidence scores
        """
        results = self.model(image, verbose=False)

        crosswalks = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for crosswalk class (you'll need to adjust class ID)
                    # For now, using a generic approach - you may need to train custom model
                    confidence = box.conf.cpu().numpy()[0]
                    if confidence >= confidence_threshold:
                        coords = box.xyxy.cpu().numpy()[0]
                        crosswalks.append({
                            'bbox': coords,  # [x1, y1, x2, y2]
                            'confidence': confidence,
                            'center': self._get_bbox_center(coords)
                        })

        return crosswalks

    def _get_bbox_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)

    def detect_crosswalk_lines(self, image: np.ndarray, roi: Optional[Tuple] = None) -> List[np.ndarray]:
        """
        Alternative method: Detect crosswalk using line detection
        Useful as backup or for more precise crosswalk line detection

        Args:
            image: Input image
            roi: Region of interest (x1, y1, x2, y2)

        Returns:
            List of detected line segments
        """
        if roi:
            x1, y1, x2, y2 = roi
            image = image[int(y1):int(y2), int(x1):int(x2)]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhanced edge detection for crosswalk stripes
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )

        if lines is not None:
            # Filter parallel lines that could be crosswalk stripes
            filtered_lines = self._filter_crosswalk_lines(lines)
            return filtered_lines

        return []

    def _filter_crosswalk_lines(self, lines: np.ndarray) -> List[np.ndarray]:
        """Filter lines to identify crosswalk stripes"""
        if len(lines) < 2:
            return []

        # Group lines by angle
        angle_threshold = 10  # degrees
        grouped_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = abs(angle)

            # Look for roughly parallel lines (crosswalk stripes)
            added_to_group = False
            for group in grouped_lines:
                group_angle = group['angle']
                if abs(angle - group_angle) < angle_threshold:
                    group['lines'].append(line[0])
                    added_to_group = True
                    break

            if not added_to_group:
                grouped_lines.append({
                    'angle': angle,
                    'lines': [line[0]]
                })

        # Return the group with most lines (likely crosswalk)
        if grouped_lines:
            best_group = max(grouped_lines, key=lambda g: len(g['lines']))
            if len(best_group['lines']) >= 3:  # At least 3 parallel lines
                return best_group['lines']

        return []

    def apply_coco_dataset(self, output_dir: str, train_split: float = 0.8):
        """
        Convert loaded COCO dataset to YOLO format for training.

        Args:
            output_dir: Directory to save converted dataset
            train_split: Ratio for train/validation split
        """
        if not self.coco_loader:
            print("No COCO dataset loaded. Initialize with coco_dataset_path parameter.")
            return

        print("Converting COCO dataset to YOLO format...")
        self.coco_loader.convert_to_yolo_format(output_dir, train_split)

    def get_coco_training_data(self) -> Optional[List]:
        """
        Get training data from COCO dataset for model enhancement.

        Returns:
            List of image paths with annotations or None
        """
        if not self.coco_loader:
            return None

        return self.coco_loader.get_all_target_images()

    def train_with_coco_data(self, output_model_path: str = "crosswalk_model.pt",
                           epochs: int = 100, batch_size: int = 16):
        """
        Train or fine-tune the model using COCO dataset.

        Args:
            output_model_path: Path to save trained model
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if not self.coco_loader:
            print("No COCO dataset loaded for training")
            return

        # Convert COCO to YOLO format first
        temp_dataset_dir = "temp_coco_yolo_dataset"
        self.apply_coco_dataset(temp_dataset_dir)

        # Train the model
        try:
            dataset_config = f"{temp_dataset_dir}/dataset.yaml"
            if Path(dataset_config).exists():
                print(f"Training model with COCO data...")
                results = self.model.train(
                    data=dataset_config,
                    epochs=epochs,
                    batch=batch_size,
                    name="crosswalk_training",
                    patience=20
                )

                # Save the trained model
                self.model.save(output_model_path)
                print(f"Model trained and saved to: {output_model_path}")
                return results
            else:
                print(f"Dataset configuration not found: {dataset_config}")
        except Exception as e:
            print(f"Error during training: {e}")

        return None