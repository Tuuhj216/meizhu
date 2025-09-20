#!/usr/bin/env python3
import numpy as np
import cv2
import time

def simulate_model_output():
    """Simulate TensorFlow Lite model output for testing"""
    # Simulate YOLO detection output: [1, 39, 8400]
    # 39 = 4 (bbox) + 1 (objectness) + 34 (classes)
    output = np.random.random((1, 39, 8400)).astype(np.float32)

    # Add some realistic detection-like values
    output[0, 4, :] = np.random.random(8400) * 0.3  # Low objectness scores
    output[0, 4, :100] = np.random.random(100) * 0.9 + 0.1  # Some higher scores

    return output

def postprocess_output(output_data):
    """Post-process model output - same as original script"""
    print(f"DEBUG: Processing output with shape: {output_data.shape}")

    if len(output_data.shape) == 3 and output_data.shape[1] == 39:  # YOLO detection format
        detections = output_data[0].T  # Shape: [8400, 39]

        # Extract components
        boxes = detections[:, :4]  # x, y, w, h
        objectness = detections[:, 4]  # objectness score
        class_probs = detections[:, 5:]  # class probabilities

        # Find detections with confidence > threshold
        confidence_threshold = 0.5
        max_class_probs = np.max(class_probs, axis=1)
        confidences = objectness * max_class_probs
        valid_detections = confidences > confidence_threshold

        if np.any(valid_detections):
            valid_boxes = boxes[valid_detections]
            valid_confidences = confidences[valid_detections]
            valid_classes = np.argmax(class_probs[valid_detections], axis=1)

            results = []
            for i in range(len(valid_boxes)):
                results.append({
                    'bbox': valid_boxes[i].tolist(),
                    'confidence': float(valid_confidences[i]),
                    'class': int(valid_classes[i])
                })

            return {
                'type': 'detection',
                'detections': results,
                'total_detections': len(results)
            }
        else:
            return {
                'type': 'detection',
                'detections': [],
                'total_detections': 0
            }

def draw_detections(frame, results):
    """Draw detection results on frame"""
    if results['type'] == 'detection' and results['detections']:
        height, width = frame.shape[:2]

        for detection in results['detections']:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class']

            # Convert normalized coordinates to pixel coordinates
            x_center, y_center, w, h = bbox
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"Class {class_id}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame

def main():
    print("Testing camera and model simulation...")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    start_time = time.time()

    print("Starting inference simulation...")
    print("Press 'q' to quit, 's' to save current frame")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Simulate inference
            model_output = simulate_model_output()

            # Process output
            results = postprocess_output(model_output)

            # Draw results
            display_frame = draw_detections(frame.copy(), results)

            # Add frame info
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            info_text = f"Frame: {frame_count}, FPS: {fps:.1f}, Detections: {results['total_detections']}"
            cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display
            cv2.imshow('Simulated Inference', display_frame)

            # Print results every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {results['total_detections']} detections, FPS: {fps:.2f}")
                if results['detections']:
                    for i, det in enumerate(results['detections'][:3]):  # Show first 3 detections
                        print(f"  Detection {i+1}: Class {det['class']}, Conf: {det['confidence']:.3f}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_output_frame_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Saved frame to {filename}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames in {elapsed:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")

if __name__ == "__main__":
    main()