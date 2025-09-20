#!/usr/bin/env python3
"""
Export segmentation model to TensorFlow Lite with int8 quantization
"""

import os
import warnings

# Set environment to disable weights_only
os.environ['TORCH_WEIGHTS_ONLY'] = 'false'

import torch

# Monkey patch torch.load to use weights_only=False
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

from ultralytics import YOLO

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    model_path = 'runs/train/crosswalk_detection6/weights/best.pt'

    print(f"Loading segmentation model: {model_path}")

    try:
        # Load the model explicitly as segmentation
        model = YOLO(model_path, task='segment')

        print("Model loaded successfully!")
        print(f"Model type: {type(model.model).__name__}")

        print("Exporting to TensorFlow Lite with int8 quantization...")

        # Export to TFLite with int8 quantization
        tflite_path = model.export(format='tflite', int8=True)

        print(f"✅ Segmentation model exported successfully!")
        print(f"📁 TFLite file: {tflite_path}")

        # Check file size
        if os.path.exists(tflite_path):
            file_size = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"📊 File size: {file_size:.2f} MB")

        return tflite_path

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n🎉 Success! Your int8 quantized segmentation TFLite model is ready!")
    else:
        print(f"\n❌ Conversion failed.")