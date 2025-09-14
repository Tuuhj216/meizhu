# Crosswalk Detection System for Blind Assistance

A YOLO-based computer vision system that helps blind people detect crosswalks and provides real-time voice guidance for proper alignment. Optimized for both desktop and embedded MPU deployment.

## Features

- **Real-time Crosswalk Detection**: Uses YOLO and computer vision to identify crosswalk patterns
- **Voice Guidance**: Audio feedback to guide users left/right for proper alignment
- **Camera Integration**: Works with USB webcams or built-in cameras
- **MPU Optimized**: Lightweight version for embedded systems and low-resource devices
- **Dual Detection Modes**: YOLO-based for accuracy, basic CV as fallback
- **Cross-platform**: Windows, Linux, macOS support

## Quick Start

### 1. Install Dependencies
```bash
# For desktop deployment
pip install -r requirements.txt

# For MPU/embedded deployment
pip install -r mpu_requirements.txt
```

### 2. Run the System
```bash
# Full version (desktop)
python main.py

# Lightweight version (MPU/embedded)
python main_mpu.py
```

### 3. Usage
- Connect your camera
- The system will automatically detect crosswalks
- Listen for voice guidance: "Move left", "Move right", or "You are aligned"
- Press 'q' to quit (if display is available)

## Training Your Own Model

### 1. Prepare Dataset
```bash
# Create dataset structure
python prepare_dataset.py

# Organize your images:
dataset/
├── images/train/    # Training images
├── images/val/      # Validation images
├── labels/train/    # Training labels (YOLO format)
└── labels/val/      # Validation labels
```

### 2. Annotate Images
Use annotation tools like:
- [LabelImg](https://github.com/tzutalin/labelImg)
- [Roboflow](https://roboflow.com)
- [CVAT](https://cvat.ai)

Label format (YOLO):
```
0 0.5 0.6 0.8 0.1    # crosswalk: class_id center_x center_y width height
1 0.3 0.65 0.1 0.02  # crosswalk_line
```

### 3. Train Model
```bash
# Start training
python train_model.py

# Validate trained model
python train_model.py --validate
```

## Deployment Options

### Option 1: Python Script
```bash
# Copy to target machine:
# - main.py (or main_mpu.py for embedded)
# - your_trained_model.pt
# - requirements.txt

pip install -r requirements.txt
python main.py
```

### Option 2: Standalone Executable
```bash
# Build executable
python build_executable.py

# Creates deployment_package/ with:
# - crosswalk_detector.exe (Windows) or crosswalk_detector (Linux/Mac)
# - Model files
# - Instructions
```

### Option 3: MPU/Embedded Systems
```bash
# Use lightweight version
python main_mpu.py

# Features:
# - Reduced resolution (320x240)
# - Frame skipping for performance
# - Fallback detection without YOLO
# - Headless operation support
```

## System Requirements

### Minimum (Basic Detection)
- Python 3.8+
- OpenCV
- NumPy
- Camera (USB webcam or built-in)

### Recommended (YOLO Detection)
- Python 3.8+
- PyTorch
- Ultralytics YOLO
- 4GB RAM
- Camera with good lighting

### MPU/Embedded
- ARM-based MPU (Raspberry Pi, etc.)
- 1GB+ RAM
- USB camera
- Optional: speakers for audio feedback

## File Structure

```
crosswalk-detection/
├── main.py                 # Main application (desktop)
├── main_mpu.py            # Optimized for MPU/embedded
├── train_model.py         # Model training script
├── prepare_dataset.py     # Dataset preparation
├── build_executable.py    # Build standalone executable
├── requirements.txt       # Desktop dependencies
├── mpu_requirements.txt   # MPU/embedded dependencies
├── deploy_requirements.txt # Deployment dependencies
├── dataset.yaml          # YOLO dataset configuration
├── deployment_guide.md   # Detailed deployment guide
├── dataset/              # Training data
│   ├── images/
│   └── labels/
└── runs/                 # Training outputs
    └── train/
        └── crosswalk_detection/
            └── weights/
                ├── best.pt    # Best trained model
                └── last.pt    # Latest checkpoint
```

## Configuration

### Camera Settings (main_mpu.py)
```python
self.camera_width = 320   # Resolution width
self.camera_height = 240  # Resolution height
self.fps_target = 10      # Target FPS
self.frame_skip = 2       # Process every Nth frame
```

### Detection Settings
```python
self.voice_cooldown = 4   # Seconds between voice messages
threshold = width * 0.15  # Alignment sensitivity
```

## Training Commands

```bash
# Basic training
python train_model.py

# Advanced training with custom parameters
yolo train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16

# Export for different formats
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx')"
```

## Performance Optimization

### For Slower Systems
- Use `main_mpu.py` instead of `main.py`
- Reduce `camera_width` and `camera_height`
- Increase `frame_skip` value
- Use basic detection mode (no YOLO)

### For Better Accuracy
- Use larger YOLO models (`yolov8m.pt`, `yolov8l.pt`)
- Increase image resolution
- Reduce `frame_skip` to 1
- Train custom model with your specific environment

## Troubleshooting

### Camera Issues
```bash
# List available cameras (Linux)
ls /dev/video*

# Test camera access
python -c "import cv2; cap=cv2.VideoCapture(0); print('Camera OK' if cap.read()[0] else 'Camera Error')"
```

### Audio Issues
```bash
# Install system audio support
# Ubuntu/Debian:
sudo apt-get install espeak espeak-data

# Test TTS
python -c "import pyttsx3; engine=pyttsx3.init(); engine.say('Test'); engine.runAndWait()"
```

### Model Loading Issues
- Ensure model file exists in project directory
- Check file permissions
- Verify PyTorch compatibility

### Memory Issues (MPU)
- Use `opencv-python-headless` instead of `opencv-python`
- Reduce batch size during training
- Use smaller YOLO model variants

## Contributing

1. Fork the repository
2. Create feature branch
3. Add your improvements
4. Test on different systems
5. Submit pull request

## License

This project is designed for assistive technology and accessibility purposes.

## Support

For issues or questions:
1. Check troubleshooting section
2. Review deployment_guide.md
3. Test with basic computer vision mode first
4. Verify camera and audio setup