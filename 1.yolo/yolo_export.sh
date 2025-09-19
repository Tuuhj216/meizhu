#!/bin/bash

# Set environment variables to handle torch loading issues
export TORCH_WEIGHTS_ONLY=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Try different models in order of preference
models=(
    "runs/train/crosswalk_detection/weights/best.pt"
    "runs/train/crosswalk_detection2/weights/best.pt"
    "runs/train/crosswalk_detection3/weights/best.pt"
)

for model in "${models[@]}"; do
    echo "Attempting to convert $model to TensorFlow Lite with int8 quantization..."

    # First try ONNX export
    echo "Step 1: Converting to ONNX..."
    python -c "
import warnings
warnings.filterwarnings('ignore')
import torch
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.SegmentationModel', 'ultralytics.nn.tasks.DetectionModel'])
from ultralytics import YOLO
model = YOLO('$model')
onnx_path = model.export(format='onnx')
print(f'ONNX exported to: {onnx_path}')
"

    if [ $? -eq 0 ]; then
        echo "ONNX export successful, now converting to TFLite..."

        # Convert to TFLite with int8
        python -c "
import warnings
warnings.filterwarnings('ignore')
import torch
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.SegmentationModel', 'ultralytics.nn.tasks.DetectionModel'])
from ultralytics import YOLO
import os
onnx_file = '${model%.pt}.onnx'
if os.path.exists(onnx_file):
    model = YOLO(onnx_file)
    tflite_path = model.export(format='tflite', int8=True)
    print(f'TFLite exported to: {tflite_path}')
    print(f'File size: {os.path.getsize(tflite_path) / (1024*1024):.2f} MB')
else:
    print(f'ONNX file not found: {onnx_file}')
"

        if [ $? -eq 0 ]; then
            echo "✅ Conversion successful!"
            break
        else
            echo "❌ TFLite conversion failed for $model"
        fi
    else
        echo "❌ ONNX export failed for $model"
    fi

    echo "---"
done