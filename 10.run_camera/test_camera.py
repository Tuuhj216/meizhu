#!/usr/bin/env python3
import cv2
import numpy as np
import time

def test_camera():
    print("Testing camera...")

    # Try different backends and configurations
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_ANY, "Any")
    ]

    for backend, name in backends:
        print(f"\nTrying {name} backend...")
        cap = cv2.VideoCapture(0, backend)

        if cap.isOpened():
            print(f"✓ {name} backend opened successfully!")

            # Set some properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Try to read a frame
            ret, frame = cap.read()

            if ret and frame is not None:
                print(f"✓ Frame captured! Shape: {frame.shape}")
                cv2.imwrite(f"test_frame_{name.lower()}.jpg", frame)
                print(f"✓ Test frame saved as test_frame_{name.lower()}.jpg")
                cap.release()
                return True
            else:
                print(f"✗ Could not read frame from {name}")
        else:
            print(f"✗ Could not open {name} backend")

        cap.release()

    print("All backends failed!")
    return False

if __name__ == "__main__":
    test_camera()