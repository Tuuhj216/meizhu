#!/usr/bin/env python3
"""
Convert YOLO PyTorch model to TensorFlow Lite with int8 quantization
"""

import os
import warnings

# Set environment variable to allow unsafe loading
os.environ['TORCH_SERIALIZATION_LEGACY'] = '1'

import torch
from ultralytics import YOLO

def convert_yolo_to_tflite(model_path, output_dir=None):
    """
    Convert YOLO .pt model to TensorFlow Lite with int8 quantization

    Args:
        model_path: Path to the .pt model file
        output_dir: Output directory (optional)
    """
    try:
        print(f"Loading model from: {model_path}")

        # Suppress warnings and load model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = YOLO(model_path)

        # Export to TensorFlow Lite with int8 quantization
        print("Exporting to TensorFlow Lite with int8 quantization...")
        exported_path = model.export(format='tflite', int8=True)

        print(f"Model exported successfully to: {exported_path}")
        return exported_path

    except Exception as e:
        print(f"Error during conversion: {e}")
        return None

if __name__ == "__main__":
    # List available models
    model_paths = [
        "runs/train/crosswalk_detection/weights/best.pt",
        "runs/train/crosswalk_detection2/weights/best.pt",
        "runs/train/crosswalk_detection3/weights/best.pt",
        "runs/train/crosswalk_detection4/weights/best.pt",
        "runs/train/crosswalk_detection5/weights/best.pt",
        "runs/train/crosswalk_detection6/weights/best.pt"
    ]

    print("Available models:")
    for i, path in enumerate(model_paths):
        if os.path.exists(path):
            print(f"{i+1}. {path}")

    # Use the latest model (crosswalk_detection6) by default
    latest_model = "runs/train/crosswalk_detection6/weights/best.pt"

    if os.path.exists(latest_model):
        print(f"\nConverting latest model: {latest_model}")
        result = convert_yolo_to_tflite(latest_model)
        if result:
            print(f"\nConversion completed! TFLite model saved at: {result}")
    else:
        print(f"Model not found: {latest_model}")