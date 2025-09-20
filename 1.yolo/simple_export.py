#!/usr/bin/env python3

import torch
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)

from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

model_path = "runs/train/crosswalk_detection6/weights/best.pt"
print(f"Converting {model_path} to TensorFlow Lite with int8 quantization...")

model = YOLO(model_path)
exported_path = model.export(format='tflite', int8=True)
print(f"Exported to: {exported_path}")