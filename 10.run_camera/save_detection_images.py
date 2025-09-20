#!/usr/bin/env python3
import numpy as np
import cv2

def create_sample_detection_image():
    """Create a sample image with detection boxes"""
    # Create a sample image (640x480)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some background pattern
    for i in range(0, 640, 20):
        cv2.line(image, (i, 0), (i, 480), (30, 30, 30), 1)
    for i in range(0, 480, 20):
        cv2.line(image, (0, i), (640, i), (30, 30, 30), 1)

    # Add some colored rectangles as "objects"
    objects = [
        {"rect": (100, 50, 150, 100), "color": (255, 100, 100)},
        {"rect": (300, 150, 200, 120), "color": (100, 255, 100)},
        {"rect": (450, 300, 120, 80), "color": (100, 100, 255)},
        {"rect": (50, 350, 180, 100), "color": (255, 255, 100)},
    ]

    for obj in objects:
        x, y, w, h = obj["rect"]
        cv2.rectangle(image, (x, y), (x+w, y+h), obj["color"], -1)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)

    return image

def draw_detection_results(image, detections):
    """Draw detection bounding boxes on image"""
    height, width = image.shape[:2]

    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class']

        # Convert normalized coordinates to pixel coordinates
        x_center, y_center, w, h = bbox
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))

        # Skip if box is too small
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        # Draw bounding box (green for detections)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"Class {class_id}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Draw label background
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)

        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return image

def main():
    print("Creating detection output images...")

    # Create sample image
    base_image = create_sample_detection_image()
    cv2.imwrite("sample_input_image.jpg", base_image)
    print("✓ Created sample input image: sample_input_image.jpg")

    # Sample detection results (from our simulation)
    sample_detections = [
        {'bbox': [0.3, 0.2, 0.2, 0.3], 'confidence': 0.85, 'class': 1},
        {'bbox': [0.6, 0.4, 0.25, 0.2], 'confidence': 0.72, 'class': 5},
        {'bbox': [0.15, 0.6, 0.3, 0.25], 'confidence': 0.91, 'class': 3},
        {'bbox': [0.75, 0.15, 0.2, 0.15], 'confidence': 0.68, 'class': 8},
    ]

    # Draw detections on image
    detection_image = draw_detection_results(base_image.copy(), sample_detections)
    cv2.imwrite("detection_output_image.jpg", detection_image)
    print("✓ Created detection output image: detection_output_image.jpg")

    # Create info overlay
    info_image = base_image.copy()
    cv2.putText(info_image, "Object Detection Results", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(info_image, f"Found {len(sample_detections)} objects", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info_image, "Green boxes = detections", (10, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Add detection info
    for i, det in enumerate(sample_detections):
        y_pos = 150 + i * 30
        text = f"Detection {i+1}: Class {det['class']}, Conf: {det['confidence']:.2f}"
        cv2.putText(info_image, text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite("detection_info_image.jpg", info_image)
    print("✓ Created info image: detection_info_image.jpg")

    # Now draw the actual detections on the info image
    final_image = draw_detection_results(info_image, sample_detections)
    cv2.imwrite("final_detection_result.jpg", final_image)
    print("✓ Created final result image: final_detection_result.jpg")

    print("\nSummary of created images:")
    print("1. sample_input_image.jpg - Original input image")
    print("2. detection_output_image.jpg - Image with detection boxes")
    print("3. detection_info_image.jpg - Image with detection information")
    print("4. final_detection_result.jpg - Complete result with boxes and info")

    print(f"\nDetection Results Summary:")
    print(f"- Total detections: {len(sample_detections)}")
    for i, det in enumerate(sample_detections):
        bbox = det['bbox']
        print(f"- Detection {i+1}: Class {det['class']}, Confidence {det['confidence']:.2f}")
        print(f"  Bounding box: center=({bbox[0]:.2f}, {bbox[1]:.2f}), size=({bbox[2]:.2f}, {bbox[3]:.2f})")

if __name__ == "__main__":
    main()