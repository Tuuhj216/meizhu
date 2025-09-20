#!/usr/bin/env python3
import numpy as np
import cv2
import time

def simulate_model_output():
    """Simulate TensorFlow Lite model output for testing"""
    output = np.random.random((1, 39, 8400)).astype(np.float32)
    output[0, 4, :] = np.random.random(8400) * 0.3  # Low objectness scores
    output[0, 4, :100] = np.random.random(100) * 0.9 + 0.1  # Some higher scores
    return output

def postprocess_output(output_data):
    """Post-process model output"""
    if len(output_data.shape) == 3 and output_data.shape[1] == 39:
        detections = output_data[0].T
        boxes = detections[:, :4]
        objectness = detections[:, 4]
        class_probs = detections[:, 5:]

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

            x_center, y_center, w, h = bbox
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame

def main():
    print("Running headless camera simulation...")

    # Try to open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera - creating synthetic frames instead")
        use_camera = False
    else:
        print("Camera opened successfully")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        use_camera = True

    frame_count = 0
    start_time = time.time()

    print("Processing 10 frames...")

    for i in range(10):
        if use_camera:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {i+1}")
                # Create a synthetic frame instead
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:
            # Create synthetic frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Synthetic Frame {i+1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Simulate inference
        model_output = simulate_model_output()
        results = postprocess_output(model_output)

        # Draw detections
        output_frame = draw_detections(frame.copy(), results)

        # Add frame info
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(output_frame, f"Frame: {frame_count}, FPS: {fps:.1f}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save frame
        filename = f"frame_{frame_count:03d}_detections.jpg"
        cv2.imwrite(filename, output_frame)

        print(f"Frame {frame_count}: {results['total_detections']} detections -> {filename}")

        if results['detections']:
            for j, det in enumerate(results['detections'][:3]):  # Show first 3
                print(f"  Detection {j+1}: Class {det['class']}, Conf: {det['confidence']:.3f}")

        time.sleep(0.1)  # Small delay

    if use_camera:
        cap.release()

    print(f"\nCompleted! Processed {frame_count} frames in {elapsed:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    print(f"Saved frames: frame_001_detections.jpg to frame_{frame_count:03d}_detections.jpg")

if __name__ == "__main__":
    main()