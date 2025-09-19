#!/usr/bin/env python3
"""
YOLO training script for crosswalk detection model.
"""

from ultralytics import YOLO
import torch
import os

i = 0 # dataset

def train_crosswalk_model():
    """Train YOLO model for crosswalk detection."""

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pretrained YOLOv8 model
    #model = YOLO('yolov8n.pt')  # nano model (fastest)
    model = YOLO('best.pt')
    # Alternative models:
    # model = YOLO('yolov8s.pt')  # small
    # model = YOLO('yolov8m.pt')  # medium
    # model = YOLO('yolov8l.pt')  # large

    # Train the model
    results = model.train(
        data=f'C:/Users/ysann/Desktop/meizu/1.yolo/dataset_k_fold/fold_{i}/dataset.yaml',           # path to dataset config
        epochs=100,                    # number of epochs
        imgsz=640,                     # image size
        batch=8,                      # batch size (adjust based on GPU memory)
        device=device,                 # training device
        workers=4,                     # number of worker threads
        project='runs/train',          # project name
        name='crosswalk_detection',    # experiment name
        save=True,                     # save checkpoints
        save_period=10,                # save every N epochs
        val=True,                      # validate during training
        plots=True,                    # save training plots
        verbose=True,                  # verbose output
        patience=50,                   # early stopping patience
        
        lr0=0.005,
        box=7.5,
        cls=1,
        dfl=1.5,

        # Data augmentation
        hsv_h=0.015,                   # HSV hue augmentation
        hsv_s=0.7,                     # HSV saturation augmentation
        hsv_v=0.4,                     # HSV value augmentation
        degrees=10.0,                  # rotation degrees
        translate=0.1,                 # translation
        scale=0.5,                     # scaling
        shear=0.0,                     # shear
        perspective=0.0,               # perspective
        flipud=0.0,                    # flip up-down
        fliplr=0.5,                    # flip left-right
        mosaic=1.0,                    # mosaic augmentation
        mixup=0.0,                     # mixup augmentation
    )

    # Export model to different formats for deployment
    model.export(format='onnx')        # ONNX format
    #model.export(format='engine')      # TensorRT (if available)

    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Last model saved at: {results.save_dir}/weights/last.pt")

    return results

def validate_model(model_path: str = None):
    """Validate the trained model."""
    if model_path is None:
        model_path = 'runs/train/crosswalk_detection/weights/best.pt'

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = YOLO(model_path)

    # Validate on test set
    metrics = model.val(data=f'C:/Users/ysann/Desktop/meizu/1.yolo/dataset_k_fold/fold_{i}/dataset.yaml')

    print(f"Validation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")

    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLO crosswalk detection model')
    parser.add_argument('--validate', action='store_true', help='Validate existing model')
    parser.add_argument('--model', type=str, help='Path to model for validation')

    args = parser.parse_args()

    if args.validate:
        validate_model(args.model)
    else:
        print("Starting YOLO crosswalk detection training...")
        train_crosswalk_model()