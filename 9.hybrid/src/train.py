import os
import yaml
import torch
from ultralytics import YOLO
import argparse
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_yolo_config(config):
    """Create YOLO dataset configuration file"""
    yolo_config = {
        'path': str(Path(config['data']['train_images']).parent.parent),
        'train': config['data']['train_images'],
        'val': config['data']['val_images'],
        'nc': config['data']['num_classes'],
        'names': ['crosswalk']
    }

    config_path = 'config/dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(yolo_config, f)

    return config_path


def train_model(config_path):
    config = load_config(config_path)

    # Create output directories
    os.makedirs(config['output']['model_save_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)

    # Create YOLO dataset config
    dataset_config = create_yolo_config(config)

    # Load model
    model = YOLO(f"{config['model']['name']}.pt")

    # Training parameters
    train_params = {
        'data': dataset_config,
        'epochs': config['training']['epochs'],
        'imgsz': config['model']['img_size'],
        'batch': config['training']['batch_size'],
        'lr0': config['training']['learning_rate'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_epochs': config['training']['warmup_epochs'],
        'patience': config['training']['patience'],
        'device': config['device'],
        'workers': config['workers'],
        'project': config['output']['results_dir'],
        'name': 'crosswalk_detection',
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': True
    }

    # Start training
    print(f"Starting training with {config['model']['name']} model...")
    results = model.train(**train_params)

    # Save final model
    model_save_path = os.path.join(config['output']['model_save_dir'], 'best_crosswalk_model.pt')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train crosswalk detection model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found!")
        return

    # Train model
    results = train_model(args.config)
    print("Training completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()