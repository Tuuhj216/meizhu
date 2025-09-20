#!/usr/bin/env python3
import numpy as np
import cv2

print("Quick Detection Demo")
print("===================")

# Create a simple test image
image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(image, (100, 100), (300, 250), (100, 150, 200), -1)
cv2.circle(image, (450, 150), 50, (200, 100, 150), -1)
cv2.putText(image, "Test Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Simulate detections
detections = [
    {'bbox': [0.3, 0.3, 0.3, 0.3], 'confidence': 0.85, 'class': 1},
    {'bbox': [0.7, 0.3, 0.15, 0.2], 'confidence': 0.72, 'class': 5},
]

# Draw detections
height, width = image.shape[:2]
for i, det in enumerate(detections):
    bbox = det['bbox']
    x_center, y_center, w, h = bbox
    x1 = int((x_center - w/2) * width)
    y1 = int((y_center - h/2) * height)
    x2 = int((x_center + w/2) * width)
    y2 = int((y_center + h/2) * height)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Class {det['class']}: {det['confidence']:.2f}"
    cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save result
cv2.imwrite("quick_detection_demo.jpg", image)
print("✓ Created quick_detection_demo.jpg")
print(f"✓ Found {len(detections)} detections")
for i, det in enumerate(detections):
    print(f"  Detection {i+1}: Class {det['class']}, Confidence {det['confidence']}")

print("\nDemo complete! Check quick_detection_demo.jpg")