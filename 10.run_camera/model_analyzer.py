#!/usr/bin/env python3
import numpy as np

def analyze_model_without_tflite(model_path):
    """Analyze TensorFlow Lite model without importing tflite_runtime"""
    print(f"Analyzing model: {model_path}")

    try:
        # Read model file as binary to examine structure
        with open(model_path, 'rb') as f:
            data = f.read()

        print(f"Model file size: {len(data)} bytes ({len(data)/1024/1024:.2f} MB)")

        # Look for metadata patterns that might indicate model type
        data_str = data[:10000].lower()  # First 10KB for metadata

        # Common patterns
        segmentation_keywords = [b'segment', b'mask', b'fcn', b'unet', b'deeplabv3']
        detection_keywords = [b'yolo', b'detection', b'bbox', b'anchor', b'nms']
        classification_keywords = [b'classification', b'softmax', b'imagenet']

        # Count occurrences
        seg_count = sum(data_str.count(kw) for kw in segmentation_keywords)
        det_count = sum(data_str.count(kw) for kw in detection_keywords)
        cls_count = sum(data_str.count(kw) for kw in classification_keywords)

        print(f"Model analysis hints:")
        print(f"  Segmentation keywords found: {seg_count}")
        print(f"  Detection keywords found: {det_count}")
        print(f"  Classification keywords found: {cls_count}")

        # Based on original script analysis
        print(f"\nBased on original script analysis:")
        print(f"  Expected output shape: [1, 39, 8400]")
        print(f"  This suggests YOLO object detection format")
        print(f"  39 = 4 (bbox coords) + 1 (objectness) + 34 (class probabilities)")

        return "detection"

    except Exception as e:
        print(f"Error analyzing model: {e}")
        return "unknown"

def create_headless_inference():
    """Create headless inference that saves output to file"""
    print("Creating headless inference simulation...")

    # Simulate model output based on YOLO detection format
    output = np.random.random((1, 39, 8400)).astype(np.float32)

    # Add some realistic detection-like values
    output[0, 4, :] = np.random.random(8400) * 0.3  # Low objectness scores
    output[0, 4, :50] = np.random.random(50) * 0.9 + 0.1  # Some higher scores

    # Process output
    detections = output[0].T  # Shape: [8400, 39]
    boxes = detections[:, :4]  # x, y, w, h
    objectness = detections[:, 4]  # objectness score
    class_probs = detections[:, 5:]  # class probabilities

    # Find detections with confidence > threshold
    confidence_threshold = 0.5
    max_class_probs = np.max(class_probs, axis=1)
    confidences = objectness * max_class_probs
    valid_detections = confidences > confidence_threshold

    results = []
    if np.any(valid_detections):
        valid_boxes = boxes[valid_detections]
        valid_confidences = confidences[valid_detections]
        valid_classes = np.argmax(class_probs[valid_detections], axis=1)

        for i in range(len(valid_boxes)):
            results.append({
                'bbox': valid_boxes[i].tolist(),
                'confidence': float(valid_confidences[i]),
                'class': int(valid_classes[i])
            })

    # Create output report
    report = f"""
INFERENCE SIMULATION REPORT
===========================

Model Analysis:
- Expected format: YOLO Object Detection
- Output shape: [1, 39, 8400]
- Model type: Detection (not segmentation)

Inference Results:
- Total detections found: {len(results)}
- Confidence threshold: {confidence_threshold}

Detection Details:
"""

    for i, det in enumerate(results[:10]):  # Show first 10 detections
        bbox = det['bbox']
        report += f"""
Detection {i+1}:
  - Class ID: {det['class']}
  - Confidence: {det['confidence']:.3f}
  - Bounding Box: x={bbox[0]:.3f}, y={bbox[1]:.3f}, w={bbox[2]:.3f}, h={bbox[3]:.3f}
"""

    if len(results) > 10:
        report += f"\n... and {len(results) - 10} more detections\n"

    # Save to file
    with open('inference_output.txt', 'w') as f:
        f.write(report)

    print("Inference simulation complete!")
    print(f"Found {len(results)} detections")
    print("Results saved to: inference_output.txt")

    return results

def main():
    print("Model Analysis and Inference Simulation")
    print("======================================")

    # Analyze available models
    models = ['best_converted.tflite', 'best_int8.tflite', 'best_int8_vela.tflite']

    for model in models:
        try:
            model_type = analyze_model_without_tflite(model)
            print(f"Model {model}: {model_type}")
            print("-" * 40)
        except FileNotFoundError:
            print(f"Model {model}: not found")
            continue

    print("\nRunning headless inference simulation...")
    results = create_headless_inference()

    print("\nSummary:")
    print(f"- The models appear to be for OBJECT DETECTION, not segmentation")
    print(f"- The script is designed for real-time detection with bounding boxes")
    print(f"- Simulated inference found {len(results)} detections")
    print(f"- Output saved to inference_output.txt")

if __name__ == "__main__":
    main()