#!/usr/bin/env python3

import cv2
import time

def test_camera_backends():
    """Test different camera backends and formats"""

    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_ANY, "Default")
    ]

    formats = [
        (cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'), "YUYV"),
        (cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), "MJPG"),
        (cv2.VideoWriter_fourcc('R', 'G', 'B', '3'), "RGB3"),
        (cv2.VideoWriter_fourcc('B', 'G', 'R', '3'), "BGR3"),
    ]

    for backend_id, backend_name in backends:
        print(f"\n=== Testing {backend_name} backend ===")

        try:
            cap = cv2.VideoCapture(0, backend_id)

            if not cap.isOpened():
                print(f"❌ Could not open camera with {backend_name}")
                continue

            print(f"✅ Camera opened with {backend_name}")

            # Test different formats
            for fourcc, format_name in formats:
                print(f"\n--- Testing {format_name} format ---")

                # Set format and resolution
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Try to read a frame with timeout
                print("Attempting to read frame...")
                start_time = time.time()

                for attempt in range(3):
                    ret, frame = cap.read()
                    elapsed = time.time() - start_time

                    if ret and frame is not None:
                        print(f"✅ Frame read successful in {elapsed:.2f}s (attempt {attempt+1})")
                        print(f"   Frame shape: {frame.shape}")
                        break
                    else:
                        print(f"❌ Frame read failed (attempt {attempt+1})")

                    if elapsed > 10:  # 10 second timeout per format
                        print("⏰ Timeout reached")
                        break

                if elapsed > 10:
                    break  # Skip other formats if this one times out

            cap.release()

        except Exception as e:
            print(f"❌ Error with {backend_name}: {e}")

if __name__ == "__main__":
    print("Camera Backend and Format Testing")
    print("=" * 40)
    test_camera_backends()