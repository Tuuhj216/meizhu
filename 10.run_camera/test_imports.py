#!/usr/bin/env python3

print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy imported successfully")
    print(f"  numpy version: {np.__version__}")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import cv2
    print("✓ cv2 imported successfully")
    print(f"  opencv version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ cv2 import failed: {e}")

try:
    import tflite_runtime.interpreter as tflite
    print("✓ tflite_runtime imported successfully")
except ImportError as e:
    print(f"✗ tflite_runtime import failed: {e}")
    try:
        import tensorflow as tf
        print("✓ tensorflow imported successfully")
        print(f"  tensorflow version: {tf.__version__}")
    except ImportError as e2:
        print(f"✗ tensorflow import also failed: {e2}")

print("Import test complete.")