#!/usr/bin/env python3
import cv2
import numpy as np

def test_gstreamer_opencv():
    print("Testing camera with GStreamer pipeline...")

    # GStreamer pipeline for camera
    gst_pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink"
    )

    print(f"Using pipeline: {gst_pipeline}")

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open camera with GStreamer pipeline")
        return False

    print("✓ Camera opened with GStreamer pipeline!")

    # Try to read a frame
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Could not read frame")
        cap.release()
        return False

    print(f"✓ Frame captured! Shape: {frame.shape}")

    # Save test image
    cv2.imwrite("test_frame_gstreamer.jpg", frame)
    print("✓ Test frame saved as test_frame_gstreamer.jpg")

    cap.release()
    print("✓ Camera test completed successfully!")
    return True

if __name__ == "__main__":
    test_gstreamer_opencv()