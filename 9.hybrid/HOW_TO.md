# HOW TO: Train Crosswalk Detection with COCO Dataset

This guide explains how to train your crosswalk detection model using COCO-style datasets.

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. COCO Dataset Requirements
Your COCO dataset should have:
- **Annotations file**: JSON file in COCO format
- **Images directory**: Contains all training images
- **Crosswalk category**: Annotations must include "crosswalk" class
- **Segmentation data**: Polygon or RLE format for crosswalk boundaries

## Training Methods

### Method 1: Direct Training (Recommended)
Train directly from COCO annotations:

```bash
python main.py --mode coco-train \
    --coco-annotations path/to/annotations.json \
    --output-model crosswalk_model.pt
```

**Parameters:**
- `--coco-annotations`: Path to COCO JSON file
- `--output-model`: Name for trained model (default: crosswalk_coco_model.pt)

### Method 2: Convert Then Train
First convert COCO to YOLO format, then train:

```bash
# Step 1: Convert COCO to YOLO format
python main.py --mode coco-convert \
    --coco-annotations annotations.json \
    --coco-images images/ \
    --output-dir yolo_dataset/

# Step 2: Train with ultralytics directly
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(
    data='yolo_dataset/dataset.yaml',
    epochs=100,
    batch=16,
    patience=20
)
model.save('trained_crosswalk_model.pt')
"
```

### Method 3: Custom Training Script
Create a custom training script:

```python
# train_crosswalk.py
from crosswalk_detector import CrosswalkDetector

# Initialize detector with COCO data
detector = CrosswalkDetector(
    model_path='yolov8n.pt',
    coco_dataset_path='path/to/annotations.json'
)

# Train model with custom parameters
results = detector.train_with_coco_data(
    output_model_path="crosswalk_model.pt",
    epochs=100,        # Number of training epochs
    batch_size=16      # Batch size for training
)

print(f"Training completed: {results}")
```

## Training Parameters

### Key Parameters You Can Adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Number of training cycles |
| `batch_size` | 16 | Training batch size |
| `patience` | 20 | Early stopping patience |
| `train_split` | 0.8 | Train/validation split ratio |

### Advanced Training Options:
```python
# For more control over training
detector = CrosswalkDetector('yolov8n.pt', 'annotations.json')

# Custom training with more parameters
results = detector.model.train(
    data='converted_dataset/dataset.yaml',
    epochs=200,
    batch=32,
    imgsz=640,
    patience=30,
    save_period=10,     # Save checkpoint every 10 epochs
    workers=8,          # Number of data loading workers
    optimizer='AdamW',  # Optimizer choice
    lr0=0.01,          # Initial learning rate
    weight_decay=0.0005 # Weight decay
)
```

## Dataset Preparation

### COCO Annotation Format Example:
```json
{
  "categories": [
    {
      "id": 1,
      "name": "crosswalk",
      "supercategory": "road"
    }
  ],
  "images": [
    {
      "id": 1,
      "width": 1920,
      "height": 1080,
      "file_name": "crosswalk_001.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, x3, y3, ...]],
      "bbox": [x, y, width, height],
      "area": 50000
    }
  ]
}
```

### Directory Structure:
```
your_dataset/
├── annotations.json
└── images/
    ├── crosswalk_001.jpg
    ├── crosswalk_002.jpg
    └── ...
```

## Using the Trained Model

### 1. Test with Single Image:
```bash
python main.py --mode image \
    --model crosswalk_model.pt \
    --image test_image.jpg
```

### 2. Real-time Detection:
```bash
python main.py --mode camera \
    --model crosswalk_model.pt
```

### 3. Video Processing:
```bash
python main.py --mode video \
    --model crosswalk_model.pt \
    --video input_video.mp4
```

## Troubleshooting

### Common Issues:

1. **Import Error for pycocotools:**
   ```bash
   pip install pycocotools
   ```

2. **CUDA Out of Memory:**
   - Reduce batch size: `--batch 8`
   - Use smaller image size: `--imgsz 416`

3. **No Crosswalk Category Found:**
   - Check your COCO annotations contain "crosswalk" category
   - Verify category names match exactly

4. **Poor Training Results:**
   - Increase epochs: `epochs=200`
   - Check data quality and annotation accuracy
   - Ensure sufficient training data (>1000 images recommended)

### Monitoring Training:
Training results are saved in `runs/detect/crosswalk_training/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Final epoch weights
- Validation metrics and plots

## Performance Tips

1. **Data Quality:**
   - Ensure accurate polygon annotations
   - Include diverse crosswalk types and conditions
   - Balance dataset across different environments

2. **Training Optimization:**
   - Start with pre-trained YOLOv8 weights
   - Use data augmentation (built into YOLOv8)
   - Monitor validation loss to avoid overfitting

3. **Hardware:**
   - Use GPU for faster training
   - Increase batch size if you have more VRAM
   - Use multiple workers for data loading

## Testing Integration

Run the test suite to verify everything works:
```bash
python test_coco_integration.py
```

This will test:
- COCO dataset loading
- YOLO conversion
- Model integration
- Command-line interface

---

For additional help or issues, check the project's main README.md or create an issue in the repository.