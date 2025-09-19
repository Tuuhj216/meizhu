#!/usr/bin/env python3
"""
Test script for COCO dataset integration with crosswalk detection system.
"""

import os
import sys
import json
from pathlib import Path

def create_sample_coco_annotation():
    """Create a sample COCO annotation file for testing."""

    sample_coco = {
        "info": {
            "description": "Sample COCO dataset for crosswalk detection",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [
            {
                "id": 1,
                "name": "Sample License",
                "url": "http://example.com/license"
            }
        ],
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
                "width": 640,
                "height": 480,
                "file_name": "sample_crosswalk.jpg",
                "license": 1
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[
                    100, 200, 540, 200, 540, 280, 100, 280
                ]],
                "area": 35200,
                "bbox": [100, 200, 440, 80],
                "iscrowd": 0
            }
        ]
    }

    return sample_coco

def test_coco_loader():
    """Test the COCO dataset loader functionality."""
    print("Testing COCO Dataset Loader...")

    try:
        from coco_dataset_loader import COCODatasetLoader

        # Create sample annotation file
        sample_coco = create_sample_coco_annotation()
        sample_annotations_path = "sample_coco_annotations.json"

        with open(sample_annotations_path, 'w') as f:
            json.dump(sample_coco, f, indent=2)

        print(f"Created sample COCO annotations: {sample_annotations_path}")

        # Create sample images directory
        sample_images_dir = "sample_coco_images"
        Path(sample_images_dir).mkdir(exist_ok=True)

        # Test loading
        loader = COCODatasetLoader(sample_annotations_path, sample_images_dir)

        if loader.coco is not None:
            print("✓ COCO annotations loaded successfully")

            # Test getting target images
            target_images = loader.get_all_target_images()
            print(f"✓ Found {len(target_images)} target images")

            # Test annotation retrieval
            if target_images:
                image_id = target_images[0]['id']
                annotations = loader.get_image_annotations(image_id)
                print(f"✓ Found {len(annotations)} annotations for image {image_id}")

                # Test YOLO conversion
                if annotations:
                    ann = annotations[0]
                    img_width = target_images[0]['width']
                    img_height = target_images[0]['height']

                    yolo_polygon = loader.coco_to_yolo_polygon(ann, img_width, img_height)

                    if yolo_polygon:
                        print(f"✓ YOLO conversion successful: {len(yolo_polygon)//2} points")
                    else:
                        print("✗ YOLO conversion failed")
        else:
            print("✗ Failed to load COCO annotations")

        # Cleanup
        os.remove(sample_annotations_path)
        os.rmdir(sample_images_dir)

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure pycocotools is installed: pip install pycocotools")
    except Exception as e:
        print(f"✗ Test failed: {e}")

def test_crosswalk_detector_integration():
    """Test the integration with CrosswalkDetector."""
    print("\nTesting CrosswalkDetector Integration...")

    try:
        from crosswalk_detector import CrosswalkDetector

        # Test creating detector without COCO
        detector = CrosswalkDetector()
        print("✓ CrosswalkDetector initialized without COCO")

        # Test methods exist
        if hasattr(detector, 'apply_coco_dataset'):
            print("✓ apply_coco_dataset method exists")
        else:
            print("✗ apply_coco_dataset method missing")

        if hasattr(detector, 'train_with_coco_data'):
            print("✓ train_with_coco_data method exists")
        else:
            print("✗ train_with_coco_data method missing")

    except ImportError as e:
        print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Test failed: {e}")

def test_main_app_integration():
    """Test the main application integration."""
    print("\nTesting Main Application Integration...")

    try:
        from main import CrosswalkNavigationApp

        # Test creating app without COCO
        app = CrosswalkNavigationApp()
        print("✓ CrosswalkNavigationApp initialized")

        # Test methods exist
        if hasattr(app, 'apply_coco_dataset'):
            print("✓ apply_coco_dataset method exists")
        else:
            print("✗ apply_coco_dataset method missing")

        if hasattr(app, 'train_with_coco'):
            print("✓ train_with_coco method exists")
        else:
            print("✗ train_with_coco method missing")

    except ImportError as e:
        print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Test failed: {e}")

def test_command_line_interface():
    """Test command line interface."""
    print("\nTesting Command Line Interface...")

    try:
        # Test help message includes COCO options
        import subprocess
        result = subprocess.run([sys.executable, "main.py", "--help"],
                              capture_output=True, text=True)

        if "coco-convert" in result.stdout:
            print("✓ COCO convert mode available")
        else:
            print("✗ COCO convert mode not found in help")

        if "coco-train" in result.stdout:
            print("✓ COCO train mode available")
        else:
            print("✗ COCO train mode not found in help")

        if "--coco-annotations" in result.stdout:
            print("✓ COCO annotations argument available")
        else:
            print("✗ COCO annotations argument not found")

    except Exception as e:
        print(f"✗ CLI test failed: {e}")

def main():
    """Run all tests."""
    print("=" * 50)
    print("COCO Dataset Integration Tests")
    print("=" * 50)

    test_coco_loader()
    test_crosswalk_detector_integration()
    test_main_app_integration()
    test_command_line_interface()

    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)

    print("\nUsage examples:")
    print("1. Convert COCO to YOLO format:")
    print("   python main.py --mode coco-convert --coco-annotations annotations.json --coco-images images/ --output-dir yolo_dataset/")

    print("\n2. Train with COCO data:")
    print("   python main.py --mode coco-train --coco-annotations annotations.json --output-model custom_crosswalk_model.pt")

    print("\n3. Use with existing functionality:")
    print("   python main.py --mode camera --coco-annotations annotations.json")

if __name__ == "__main__":
    main()