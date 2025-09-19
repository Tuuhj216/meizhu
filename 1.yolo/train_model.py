#!/usr/bin/env python3
"""
YOLO training script for crosswalk detection model.
"""

from ultralytics import YOLO
import torch
import os
import gc
import time

# Monkey patch torch.load to use weights_only=False for YOLO models
original_torch_load = torch.load


def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)


torch.load = patched_torch_load


def cleanup_resources():
    """Comprehensive resource cleanup function."""
    print("Performing resource cleanup...")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"CUDA memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()}")

    # Force garbage collection
    gc.collect()

    # Brief pause for system cleanup
    time.sleep(3)
    print("Resource cleanup completed.")


# i = 4  # dataset


def train_crosswalk_model(i: int, previous_best_model_path: str):
    """Train YOLO model for crosswalk detection."""

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO('yolov8m-seg.pt')

    if i != 0:
        # model = YOLO(f'runs/train/crosswalk_detection{i-1}/weights/best.pt')
        print("Loading previous best model for fine-tuning... from: ", previous_best_model_path)
        model = YOLO(previous_best_model_path)

    # Alternative models:
    # model = YOLO('yolov8n-seg.pt')  # nano
    # model = YOLO('yolov8s-seg.pt')  # small
    # model = YOLO('yolov8l-seg.pt')  # large

    # Train the model
    results = model.train(
        # path to dataset config
        data=os.path.abspath(f'dataset_k_fold/fold_{i}/dataset.yaml'),
        epochs=100,                    # number of epochs
        imgsz=640,                      # balanced resolution for speed
        batch=20,                       # larger batch for faster training
        device=device,                 # training device
        workers=32,                    # 32 workers for RTX 5070 Ti
        project='runs/train',          # project name
        name='crosswalk_detection',    # experiment name
        save=True,                     # save checkpoints
        save_period=25,                # save less frequently for speed
        val=True,                      # validate during training
        plots=True,                    # save training plots
        verbose=True,                  # verbose output
        patience=50,                   # early stopping patience

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
    # model.export(format='engine')      # TensorRT (if available)

    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Last model saved at: {results.save_dir}/weights/last.pt")

    # Save the path before cleanup
    best_model_path = f"{results.save_dir}/weights/best.pt"

    # Clean up resources
    del model
    del results

    # Use comprehensive cleanup function
    cleanup_resources()

    return None, best_model_path


def validate_model(model_path: str = None):
    """Validate the trained model."""
    if model_path is None:
        model_path = 'runs/train/crosswalk_detection/weights/best.pt'

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = YOLO(model_path)

    # Validate on test set
    metrics = model.val(data=os.path.abspath(
        f'dataset_k_fold/fold_{i}/dataset.yaml'))

    print(f"Validation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train YOLO crosswalk detection model')
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing model')
    parser.add_argument('--model', type=str,
                        help='Path to model for validation')

    args = parser.parse_args()

    if args.validate:
        validate_model(args.model)
    else:
        print("Starting YOLO crosswalk detection training...")

        previous_best_model_path = "runs/train/crosswalk_detection2/weights/best.pt"

        for i in range(3,5):  # 5-fold cross-validation
            print(f"\n=== Starting fold {i+1}/5 ===")

            # Clear memory before each training iteration
            cleanup_resources()

            _, previous_best_model_path = train_crosswalk_model(
                i, previous_best_model_path)

            print(f"=== Completed fold {i+1}/5 ===")
            print(f"Memory cleanup completed, waiting before next fold...")

            # Additional wait between folds for complete resource release
            time.sleep(5)
