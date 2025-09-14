# Crosswalk Detection Model Training & Deployment Guide

## 1. Training Your Custom Model

### Step 1: Prepare Your Dataset
```bash
# Create dataset structure
python prepare_dataset.py

# Organize your data:
dataset/
├── images/
│   ├── train/     # Training images (.jpg, .png)
│   └── val/       # Validation images
└── labels/
    ├── train/     # Training labels (.txt files)
    └── val/       # Validation labels
```

### Step 2: Annotate Your Images
Use tools like:
- **LabelImg**: https://github.com/tzutalin/labelImg
- **Roboflow**: https://roboflow.com
- **CVAT**: https://cvat.ai

Label format (YOLO):
```
# Each line: class_id center_x center_y width height (normalized 0-1)
0 0.5 0.6 0.8 0.1    # crosswalk
1 0.3 0.65 0.1 0.02  # crosswalk line
```

### Step 3: Train the Model
```bash
# Install dependencies
pip install -r requirements.txt

# Start training
python train_model.py

# Validate trained model
python train_model.py --validate
```

Training will create:
- `runs/train/crosswalk_detection/weights/best.pt` - Best model
- `runs/train/crosswalk_detection/weights/last.pt` - Latest checkpoint

## 2. Deployment Options

### Option A: Python Script Deployment
```bash
# Copy required files to target machine
- main.py
- your_trained_model.pt (or use default yolov8n.pt)
- deploy_requirements.txt

# Install dependencies
pip install -r deploy_requirements.txt

# Run application
python main.py
```

### Option B: Standalone Executable
```bash
# Create executable (see build_executable.py)
python build_executable.py

# This creates:
dist/crosswalk_detector.exe  # Windows
dist/crosswalk_detector      # Linux/Mac
```

### Option C: Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY deploy_requirements.txt .
RUN pip install -r deploy_requirements.txt

COPY main.py your_model.pt ./
CMD ["python", "main.py"]
```

## 3. Model Optimization for Different Machines

### For Slower Machines:
```python
# In main.py, modify:
class CrosswalkDetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):  # Use nano model
        # Reduce image processing
        self.process_every_n_frames = 3  # Process every 3rd frame
```

### For Better Performance:
- Use `yolov8s.pt` or `yolov8m.pt` for better accuracy
- Enable GPU: `torch.cuda.is_available()`
- Use TensorRT: Export model with `format='engine'`

## 4. Deployment Commands

### Training Command:
```bash
# Basic training
python train_model.py

# Advanced training with custom parameters
yolo train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16
```

### Export for Different Formats:
```python
from ultralytics import YOLO

model = YOLO('runs/train/crosswalk_detection/weights/best.pt')
model.export(format='onnx')     # Cross-platform
model.export(format='engine')   # NVIDIA TensorRT
model.export(format='coreml')   # Apple devices
model.export(format='tflite')   # Mobile devices
```

### Compilation to Executable:
```bash
# Create standalone executable
pyinstaller --onefile --add-data "best.pt;." main.py

# Or use the build script
python build_executable.py
```

## 5. Performance Optimization

### CPU Optimization:
- Use smaller models (yolov8n.pt)
- Reduce image resolution
- Process fewer frames per second

### GPU Optimization:
- Use CUDA-enabled PyTorch
- Increase batch size
- Use TensorRT for inference

### Memory Optimization:
- Use opencv-python-headless (no GUI dependencies)
- Clear unused variables
- Process frames in chunks

## 6. Cross-Platform Considerations

### Windows:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python main.py
```

### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install python3-opencv espeak
pip install -r deploy_requirements.txt
```

### macOS:
```bash
brew install espeak
pip install -r deploy_requirements.txt
```

## Troubleshooting

1. **CUDA not found**: Install PyTorch with CPU support
2. **Camera not detected**: Check camera permissions and index
3. **TTS not working**: Install system speech engines (espeak/SAPI)
4. **Model not loading**: Ensure model file path is correct

## File Structure for Deployment:
```
deployment_package/
├── main.py                    # Main application
├── best.pt                    # Your trained model
├── deploy_requirements.txt    # Dependencies
├── README.md                  # Usage instructions
└── assets/                    # Optional: icons, sounds
```