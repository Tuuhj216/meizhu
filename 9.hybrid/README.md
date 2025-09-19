# Crosswalk Navigation System

A YOLO-based computer vision system that detects crosswalks and provides navigation directions based on user intent.

## Features

- **YOLO Crosswalk Detection**: Uses YOLO model to detect crosswalks in images/video
- **Vector Calculation**: Calculates crosswalk orientation and direction vectors
- **Navigation Logic**: Determines whether to turn left, right, or go straight based on crosswalk position
- **Real-time Processing**: Supports camera feed and video file processing
- **Fallback Detection**: Uses line detection as backup when YOLO detection fails

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Download YOLO model (optional - will download automatically):
```bash
# The system will use yolov8n.pt by default
# For better crosswalk detection, consider training a custom model
```

## Usage

### Command Line Interface

#### Process single image:
```bash
python main.py --mode image --image path/to/image.jpg --direction forward
```

#### Process video file:
```bash
python main.py --mode video --video path/to/video.mp4 --direction left
```

#### Real-time camera processing:
```bash
python main.py --mode camera --camera 0 --direction right
```

### Interactive Controls (Camera/Video mode)

- `q`: Quit application
- `l`: Set direction to LEFT
- `r`: Set direction to RIGHT
- `s`: Set direction to STRAIGHT

### Python API

```python
from main import CrosswalkNavigationApp

# Initialize the application
app = CrosswalkNavigationApp('yolov8n.pt')

# Process single image
results = app.process_image('image.jpg', 'forward')

# Process video frame
import cv2
image = cv2.imread('image.jpg')
results = app.process_frame(image, 'left')

# Real-time processing
app.process_video_stream(0, 'forward')  # Use camera 0
```

## System Architecture

### Components

1. **CrosswalkDetector** (`crosswalk_detector.py`)
   - YOLO-based crosswalk detection
   - Line detection fallback method
   - Confidence scoring

2. **VectorCalculator** (`vector_calculator.py`)
   - Crosswalk orientation calculation
   - Vector mathematics for direction analysis
   - Position calculations relative to user path

3. **NavigationSystem** (`navigation_system.py`)
   - Direction determination logic
   - Navigation instruction generation
   - Confidence scoring for decisions

4. **Main Application** (`main.py`)
   - Integration of all components
   - CLI interface and video processing
   - Real-time visualization

### Detection Methods

#### Primary: YOLO Detection
- Uses pre-trained YOLO model
- Returns bounding boxes with confidence scores
- Fast and reliable for general object detection

#### Fallback: Line Detection
- Canny edge detection + Hough line transform
- Identifies parallel lines characteristic of crosswalks
- More precise for crosswalk stripe detection

### Navigation Logic

The system determines navigation direction using:

1. **Crosswalk Position**: Relative to image center (camera viewpoint)
2. **Crosswalk Orientation**: Direction vector of crosswalk stripes
3. **User Intent**: Desired direction (forward, left, right)

#### Direction Classification:
- **STRAIGHT**: Crosswalk is centered and aligned with user direction
- **LEFT**: User should turn left to reach crosswalk
- **RIGHT**: User should turn right to reach crosswalk
- **TURN_AROUND**: Crosswalk is in opposite direction

## Configuration

### Model Configuration
- Default model: `yolov8n.pt` (lightweight)
- For better accuracy: Use `yolov8s.pt`, `yolov8m.pt`, or custom trained model
- Custom training recommended for specific crosswalk types

### Detection Parameters
- Confidence threshold: 0.5 (adjustable in CrosswalkDetector)
- Line detection sensitivity: Configurable in Canny/Hough parameters
- Angle threshold for turns: 30 degrees (adjustable in NavigationSystem)

## Training Custom Model

For better crosswalk detection, train a custom YOLO model:

1. Collect crosswalk images
2. Annotate with crosswalk bounding boxes
3. Train YOLO model with crosswalk class
4. Replace model path in initialization

## Limitations

- Requires good lighting conditions
- Performance depends on camera quality and angle
- May need custom training for specific crosswalk types
- Distance estimation is in pixels (could be calibrated for real units)

## Future Improvements

- [ ] Add distance estimation in real units
- [ ] Integrate with GPS for enhanced navigation
- [ ] Add audio navigation instructions
- [ ] Improve detection in low-light conditions
- [ ] Add support for different crosswalk types (zebra, painted, etc.)
- [ ] Real-time model optimization for mobile devices