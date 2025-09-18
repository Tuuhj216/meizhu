import os
import cv2
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def detect_crosswalks(model, image_path, conf_threshold=0.25, save_result=True, output_dir='results'):
    """
    Detect crosswalks in a single image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    # Run detection
    results = model(image, conf=conf_threshold)

    # Process results
    detections = []
    annotated_image = image.copy()

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()

                # Store detection
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence)
                })

                # Draw bounding box
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Add label
                label = f'Crosswalk: {confidence:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                cv2.putText(annotated_image, label, (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Save annotated image if requested
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        filename = Path(image_path).stem + '_detected.jpg'
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, annotated_image)
        print(f"Result saved to: {output_path}")

    return detections, annotated_image


def detect_in_directory(model, input_dir, conf_threshold=0.25, output_dir='results'):
    """
    Detect crosswalks in all images in a directory
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(Path(input_dir).glob(f'*{ext}'))
        image_paths.extend(Path(input_dir).glob(f'*{ext.upper()}'))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process")

    all_detections = {}

    for image_path in image_paths:
        print(f"Processing: {image_path.name}")
        detections, _ = detect_crosswalks(
            model, str(image_path), conf_threshold,
            save_result=True, output_dir=output_dir
        )
        all_detections[str(image_path)] = detections

    return all_detections


def detect_in_video(model, video_path, conf_threshold=0.25, output_path='output_video.mp4'):
    """
    Detect crosswalks in a video file
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on frame
        results = model(frame, conf=conf_threshold)

        # Annotate frame
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Add label
                    label = f'Crosswalk: {confidence:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Detect crosswalks in images or videos')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, directory of images, or video file')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')

    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        conf_threshold = config.get('validation', {}).get('conf_threshold', args.conf)
    else:
        conf_threshold = args.conf

    # Load model
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Determine source type and process
    source_path = Path(args.source)

    if source_path.is_file():
        if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video file
            output_video = os.path.join(args.output, f"{source_path.stem}_detected.mp4")
            detect_in_video(model, str(source_path), conf_threshold, output_video)
        else:
            # Single image
            detections, _ = detect_crosswalks(
                model, str(source_path), conf_threshold,
                save_result=True, output_dir=args.output
            )
            print(f"Found {len(detections)} crosswalk(s)")
    elif source_path.is_dir():
        # Directory of images
        all_detections = detect_in_directory(model, str(source_path), conf_threshold, args.output)
        total_detections = sum(len(dets) for dets in all_detections.values())
        print(f"Processed {len(all_detections)} images, found {total_detections} crosswalk(s) total")
    else:
        print(f"Source not found: {args.source}")


if __name__ == "__main__":
    main()