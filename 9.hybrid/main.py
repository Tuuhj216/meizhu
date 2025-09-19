import cv2
import numpy as np
from typing import Optional, Tuple
import argparse
import sys

from crosswalk_detector import CrosswalkDetector
from vector_calculator import VectorCalculator
from navigation_system import NavigationSystem, Direction


class CrosswalkNavigationApp:
    def __init__(self, model_path: str = 'yolov8n.pt', coco_annotations: Optional[str] = None):
        """
        Initialize the crosswalk navigation application

        Args:
            model_path: Path to YOLO model weights
            coco_annotations: Optional path to COCO annotations file for enhanced detection
        """
        self.detector = CrosswalkDetector(model_path, coco_annotations)
        self.vector_calc = VectorCalculator()
        self.navigation = NavigationSystem()

    def process_image(self, image_path: str, user_direction: str = "forward") -> dict:
        """
        Process a single image for crosswalk detection and navigation

        Args:
            image_path: Path to input image
            user_direction: User's intended direction ("forward", "left", "right")

        Returns:
            Dictionary containing detection results and navigation instructions
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return self.process_frame(image, user_direction)

    def process_frame(self, image: np.ndarray, user_direction: str = "forward") -> dict:
        """
        Process a single frame for crosswalk detection and navigation

        Args:
            image: Input image as numpy array
            user_direction: User's intended direction

        Returns:
            Dictionary containing detection results and navigation instructions
        """
        height, width = image.shape[:2]
        image_center = (width // 2, height // 2)

        # Detect crosswalks using YOLO
        crosswalks = self.detector.detect_crosswalk(image)

        if not crosswalks:
            # Fallback: try line detection method
            lines = self.detector.detect_crosswalk_lines(image)
            if lines:
                # Create a synthetic crosswalk detection from lines
                crosswalk_center = self._calculate_lines_center(lines)
                crosswalks = [{
                    'bbox': None,
                    'confidence': 0.7,  # Medium confidence for line detection
                    'center': crosswalk_center,
                    'lines': lines
                }]

        results = {
            'crosswalks_detected': len(crosswalks),
            'navigation_instructions': [],
            'image_center': image_center,
            'image_shape': (height, width)
        }

        if crosswalks:
            # Process the most confident crosswalk detection
            best_crosswalk = max(crosswalks, key=lambda x: x['confidence'])
            crosswalk_center = best_crosswalk['center']

            # Calculate crosswalk vector if possible
            crosswalk_vector = None
            if 'lines' in best_crosswalk and best_crosswalk['lines']:
                crosswalk_vector = self.vector_calc.calculate_crosswalk_vector(
                    best_crosswalk['lines'], (height, width)
                )
            elif best_crosswalk['bbox'] is not None:
                crosswalk_vector = self.vector_calc.calculate_crosswalk_vector_from_bbox(
                    best_crosswalk['bbox'], image
                )

            # Determine navigation direction
            if crosswalk_vector:
                # Advanced navigation with vector analysis
                user_dir_vector = self._get_user_direction_vector(user_direction)
                direction = self.navigation.determine_direction(
                    user_dir_vector, crosswalk_vector, crosswalk_center, image_center
                )
            else:
                # Simple position-based navigation
                direction = self.navigation.determine_direction_simple(
                    crosswalk_center, image_center, user_direction
                )

            # Generate navigation instruction
            nav_instruction = self.navigation.calculate_navigation_instruction(
                direction, crosswalk_center, image_center
            )

            results['navigation_instructions'].append(nav_instruction)
            results['best_crosswalk'] = {
                'center': crosswalk_center,
                'confidence': best_crosswalk['confidence'],
                'vector': crosswalk_vector
            }

        return results

    def process_video_stream(self, source: int = 0, user_direction: str = "forward"):
        """
        Process video stream for real-time crosswalk navigation

        Args:
            source: Video source (0 for webcam, or video file path)
            user_direction: User's intended direction
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        print("Starting crosswalk navigation system...")
        print("Press 'q' to quit, 'l' for left, 'r' for right, 's' for straight")

        current_direction = user_direction

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            try:
                results = self.process_frame(frame, current_direction)

                # Draw results on frame
                annotated_frame = self._draw_results(frame, results)

                # Display frame
                cv2.imshow('Crosswalk Navigation', annotated_frame)

                # Print navigation instructions
                if results['navigation_instructions']:
                    instruction = results['navigation_instructions'][0]
                    print(f"Direction: {instruction['direction']} - {instruction['instruction']}")

            except Exception as e:
                print(f"Error processing frame: {e}")
                annotated_frame = frame

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                current_direction = "left"
                print("Direction set to: LEFT")
            elif key == ord('r'):
                current_direction = "right"
                print("Direction set to: RIGHT")
            elif key == ord('s'):
                current_direction = "forward"
                print("Direction set to: STRAIGHT")

        cap.release()
        cv2.destroyAllWindows()

    def _get_user_direction_vector(self, direction: str) -> Tuple[float, float]:
        """Convert user direction string to vector"""
        if direction.lower() == "forward":
            return (0, -1)  # Pointing up in image coordinates
        elif direction.lower() == "left":
            return (-1, 0)  # Pointing left
        elif direction.lower() == "right":
            return (1, 0)   # Pointing right
        else:
            return (0, -1)  # Default to forward

    def _calculate_lines_center(self, lines) -> Tuple[float, float]:
        """Calculate center point of detected lines"""
        if not lines:
            return (0, 0)

        total_x = 0
        total_y = 0
        point_count = 0

        for line in lines:
            x1, y1, x2, y2 = line
            total_x += x1 + x2
            total_y += y1 + y2
            point_count += 2

        if point_count > 0:
            return (total_x / point_count, total_y / point_count)
        return (0, 0)

    def _draw_results(self, image: np.ndarray, results: dict) -> np.ndarray:
        """Draw detection results and navigation instructions on image"""
        annotated = image.copy()

        # Draw image center
        center = results['image_center']
        cv2.circle(annotated, center, 5, (0, 255, 0), -1)

        # Draw crosswalk detection
        if 'best_crosswalk' in results:
            crosswalk = results['best_crosswalk']
            center = tuple(map(int, crosswalk['center']))
            cv2.circle(annotated, center, 10, (0, 0, 255), 2)
            cv2.putText(annotated, f"Crosswalk ({crosswalk['confidence']:.2f})",
                       (center[0] + 15, center[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw navigation instruction
        if results['navigation_instructions']:
            instruction = results['navigation_instructions'][0]
            text = f"{instruction['direction'].upper()}: {instruction['instruction']}"
            cv2.putText(annotated, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw direction arrow
            self._draw_direction_arrow(annotated, instruction['direction'], center)

        return annotated

    def _draw_direction_arrow(self, image: np.ndarray, direction: str, center: Tuple[int, int]):
        """Draw direction arrow on image"""
        cx, cy = center
        arrow_length = 50

        if direction == "left":
            end_point = (cx - arrow_length, cy)
        elif direction == "right":
            end_point = (cx + arrow_length, cy)
        elif direction == "straight":
            end_point = (cx, cy - arrow_length)
        else:
            return

        cv2.arrowedLine(image, center, end_point, (255, 255, 0), 3)

    def apply_coco_dataset(self, coco_annotations: str, coco_images: str, output_dir: str):
        """
        Apply COCO dataset to the crosswalk detection system.

        Args:
            coco_annotations: Path to COCO annotations JSON file
            coco_images: Path to COCO images directory
            output_dir: Output directory for YOLO format dataset
        """
        from coco_dataset_loader import COCODatasetLoader

        loader = COCODatasetLoader(coco_annotations, coco_images, target_class="crosswalk")
        loader.convert_to_yolo_format(output_dir)
        print(f"COCO dataset converted and saved to: {output_dir}")

    def train_with_coco(self, coco_annotations: str, output_model: str = "crosswalk_coco_model.pt"):
        """
        Train the model using COCO dataset.

        Args:
            coco_annotations: Path to COCO annotations
            output_model: Output path for trained model
        """
        if hasattr(self.detector, 'coco_loader') and self.detector.coco_loader:
            results = self.detector.train_with_coco_data(output_model)
            if results:
                print(f"Training completed. Model saved as: {output_model}")
            return results
        else:
            print("No COCO dataset loaded in detector")
            return None


def main():
    parser = argparse.ArgumentParser(description="Crosswalk Navigation System")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--direction", default="forward",
                       choices=["forward", "left", "right"],
                       help="User's intended direction")
    parser.add_argument("--mode", default="camera",
                       choices=["image", "video", "camera", "coco-convert", "coco-train"],
                       help="Processing mode")
    parser.add_argument("--coco-annotations", help="Path to COCO annotations JSON file")
    parser.add_argument("--coco-images", help="Path to COCO images directory")
    parser.add_argument("--output-dir", help="Output directory for COCO conversion")
    parser.add_argument("--output-model", default="crosswalk_coco_model.pt",
                       help="Output path for trained model")

    args = parser.parse_args()

    try:
        app = CrosswalkNavigationApp(args.model, args.coco_annotations)

        if args.mode == "coco-convert":
            if args.coco_annotations and args.coco_images and args.output_dir:
                app.apply_coco_dataset(args.coco_annotations, args.coco_images, args.output_dir)
            else:
                print("For COCO conversion, provide --coco-annotations, --coco-images, and --output-dir")

        elif args.mode == "coco-train":
            if args.coco_annotations:
                app.train_with_coco(args.coco_annotations, args.output_model)
            else:
                print("For COCO training, provide --coco-annotations")

        elif args.mode == "image" and args.image:
            results = app.process_image(args.image, args.direction)
            print("Detection Results:")
            print(f"Crosswalks detected: {results['crosswalks_detected']}")
            if results['navigation_instructions']:
                instruction = results['navigation_instructions'][0]
                print(f"Navigation: {instruction['instruction']}")

        elif args.mode == "video" and args.video:
            app.process_video_stream(args.video, args.direction)

        elif args.mode == "camera":
            app.process_video_stream(args.camera, args.direction)

        else:
            print("Please specify input source with --image, --video, or use --mode camera")
            print("For COCO operations, use --mode coco-convert or --mode coco-train")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()