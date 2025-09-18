# Crosswalk Detection Project

A deep learning project for detecting crosswalks in images and videos using YOLO models with COCO dataset format.

## Project Structure

```
9.hybrid/
├── config/
│   ├── config.yaml          # Main configuration file
│   └── dataset.yaml         # Auto-generated YOLO dataset config
├── data/
│   ├── train/
│   │   ├── images/         # Training images
│   │   └── annotations.json # COCO format annotations
│   └── val/
│       ├── images/         # Validation images
│       └── annotations.json # COCO format annotations
├── models/                  # Saved model weights
├── src/
│   ├── train.py            # Training script
│   └── detect.py           # Inference script
├── utils/
│   └── dataset.py          # Dataset utilities
├── logs/                   # Training logs
├── results/                # Detection results
└── requirements.txt        # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your COCO dataset:
   - Place training images in `data/train/images/`
   - Place training annotations in `data/train/annotations.json`
   - Place validation images in `data/val/images/`
   - Place validation annotations in `data/val/annotations.json`

3. Configure the project by editing `config/config.yaml` as needed.

## Training

Train the model using:

```bash
python src/train.py --config config/config.yaml
```

The script will:
- Load your COCO dataset
- Train a YOLO model for crosswalk detection
- Save the best model to `models/best_crosswalk_model.pt`
- Generate training plots and logs

## Inference

### Single Image Detection
```bash
python src/detect.py --model models/best_crosswalk_model.pt --source path/to/image.jpg --output results/
```

### Batch Image Detection
```bash
python src/detect.py --model models/best_crosswalk_model.pt --source path/to/images/ --output results/
```

### Video Detection
```bash
python src/detect.py --model models/best_crosswalk_model.pt --source path/to/video.mp4 --output results/
```

### Options
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Output directory for results

## Configuration

Edit `config/config.yaml` to customize:
- Dataset paths
- Model type (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
- Training parameters (epochs, batch size, learning rate, etc.)
- Validation settings
- Hardware settings

## COCO Dataset Format

Your annotations should follow the COCO format with crosswalk category. Example structure:

```json
{
  "images": [...],
  "annotations": [...],
  "categories": [
    {
      "id": 1,
      "name": "crosswalk",
      "supercategory": "traffic"
    }
  ]
}
```

## Model Performance

The project supports various YOLO model sizes:
- `yolov8n`: Fastest, smallest model
- `yolov8s`: Balanced speed and accuracy
- `yolov8m`: Medium model
- `yolov8l`: Large model
- `yolov8x`: Largest, most accurate model

Choose based on your speed vs accuracy requirements.