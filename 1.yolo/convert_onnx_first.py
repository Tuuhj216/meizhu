#!/usr/bin/env python3

import torch
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)

from ultralytics import YOLO
import warnings
import os
warnings.filterwarnings('ignore')

model_path = "runs/train/crosswalk_detection6/weights/best.pt"

try:
    print(f"Step 1: Converting {model_path} to ONNX...")
    model = YOLO(model_path)
    onnx_path = model.export(format='onnx')
    print(f"ONNX exported to: {onnx_path}")

    # Check if ONNX file was created
    if os.path.exists(onnx_path):
        print(f"ONNX file size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

        print("\nStep 2: Converting ONNX to TensorFlow Lite with int8...")
        # Load from ONNX and export to TFLite
        model_onnx = YOLO(onnx_path)
        tflite_path = model_onnx.export(format='tflite', int8=True)
        print(f"TFLite exported to: {tflite_path}")

        if os.path.exists(tflite_path):
            print(f"TFLite file size: {os.path.getsize(tflite_path) / (1024*1024):.2f} MB")
            print("✅ Conversion successful!")
    else:
        print("❌ ONNX export failed")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()