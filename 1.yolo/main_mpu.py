#!/usr/bin/env python3
"""
Optimized crosswalk detection system for MPU/embedded systems.
Lightweight version with reduced resource usage and optimized inference.
"""

import cv2
import numpy as np
import time
import threading
from typing import Tuple, Optional
import os

# Try importing optimized libraries, fallback to standard ones
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available, using basic line detection only")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: TTS not available, using print output only")

class LightweightCrosswalkDetector:
    def __init__(self, model_path: str = None, camera_index: int = 0):
        """
        Initialize lightweight crosswalk detector for MPU deployment.

        Args:
            model_path: Path to YOLO model (optional for basic detection)
            camera_index: Camera device index
        """
        self.camera_index = camera_index
        self.model = None
        self.use_yolo = False

        # Initialize YOLO if available and model exists
        if YOLO_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.use_yolo = True
                print(f"✓ Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")

        # Initialize TTS if available
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 120)  # Slower for clarity
                self.tts_engine.setProperty('volume', 1.0)
            except Exception as e:
                print(f"Warning: Could not initialize TTS: {e}")

        # Performance optimization settings
        self.frame_skip = 2  # Process every 2nd frame to reduce load
        self.frame_count = 0
        self.last_voice_time = 0
        self.voice_cooldown = 4  # Longer cooldown for embedded systems

        # Camera settings for MPU optimization
        self.camera_width = 320   # Reduced resolution
        self.camera_height = 240
        self.fps_target = 10      # Lower FPS for stability

    def speak_or_print(self, text: str):
        """Output message via TTS or print (fallback for systems without audio)."""
        current_time = time.time()
        if current_time - self.last_voice_time > self.voice_cooldown:
            self.last_voice_time = current_time

            if self.tts_engine:
                try:
                    # Non-blocking TTS
                    threading.Thread(
                        target=lambda: self.tts_engine.say(text) or self.tts_engine.runAndWait(),
                        daemon=True
                    ).start()
                except Exception:
                    print(f"VOICE: {text}")
            else:
                print(f"VOICE: {text}")

    def detect_crosswalk_basic(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Basic crosswalk detection using computer vision (no YOLO required).
        Optimized for low-resource systems.
        """
        # Resize for faster processing
        height, width = frame.shape[:2]
        if width > 320:
            scale = 320 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_small = cv2.resize(frame, (new_width, new_height))
        else:
            frame_small = frame
            scale = 1.0

        # Convert to grayscale
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Edge detection with optimized parameters
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

        # Detect lines - focus on lower half of image (where crosswalks typically are)
        roi_height = edges.shape[0] // 2
        roi = edges[roi_height:, :]

        lines = cv2.HoughLinesP(
            roi, 1, np.pi/180, threshold=30,
            minLineLength=20, maxLineGap=5
        )

        if lines is None:
            return False, None

        # Filter for horizontal lines and count them
        horizontal_lines = []
        roi_width = roi.shape[1]

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Adjust y coordinates back to full frame
            y1 += roi_height
            y2 += roi_height

            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Keep roughly horizontal lines
            if angle < 25 or angle > 155:
                horizontal_lines.append([x1, y1, x2, y2])

        # Need at least 2 parallel lines for crosswalk
        if len(horizontal_lines) < 2:
            return False, None

        # Calculate average center position of detected lines
        line_centers = []
        for x1, y1, x2, y2 in horizontal_lines:
            center_x = (x1 + x2) / 2
            line_centers.append(center_x * (1/scale))  # Scale back to original

        avg_center = np.mean(line_centers)
        frame_center = width / 2

        # Alignment guidance with wider threshold for stability
        threshold = width * 0.15

        if avg_center < frame_center - threshold:
            return True, "Move right to align with crosswalk"
        elif avg_center > frame_center + threshold:
            return True, "Move left to align with crosswalk"
        else:
            return True, "Crosswalk detected, you are aligned"

    def detect_crosswalk_yolo(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        """YOLO-based crosswalk detection (when model is available)."""
        try:
            results = self.model(frame, conf=0.5, verbose=False)

            if results and len(results[0].boxes) > 0:
                # Get the first detected crosswalk
                box = results[0].boxes[0]
                x_center = float(box.xywh[0][0])
                frame_center = frame.shape[1] / 2
                threshold = frame.shape[1] * 0.15

                if x_center < frame_center - threshold:
                    return True, "Move right to align with crosswalk"
                elif x_center > frame_center + threshold:
                    return True, "Move left to align with crosswalk"
                else:
                    return True, "Crosswalk detected, you are aligned"

            return False, None

        except Exception as e:
            print(f"YOLO detection error: {e}")
            # Fallback to basic detection
            return self.detect_crosswalk_basic(frame)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with optimized detection."""
        self.frame_count += 1

        # Skip frames to reduce processing load
        if self.frame_count % self.frame_skip != 0:
            return frame

        # Choose detection method
        if self.use_yolo and self.model:
            has_crosswalk, guidance = self.detect_crosswalk_yolo(frame)
        else:
            has_crosswalk, guidance = self.detect_crosswalk_basic(frame)

        # Minimal visual feedback for MPU (reduced text rendering)
        if has_crosswalk:
            cv2.circle(frame, (20, 20), 10, (0, 255, 0), -1)  # Green dot
            if guidance:
                self.speak_or_print(guidance)
        else:
            cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)  # Red dot

        return frame

    def run_camera(self):
        """Run camera with MPU-optimized settings."""
        # Initialize camera with optimized settings
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return

        # Set camera properties for better performance on MPU
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag

        print(f"Camera initialized: {self.camera_width}x{self.camera_height} @ {self.fps_target}fps")
        self.speak_or_print("Crosswalk detection system started")

        try:
            frame_time = 1.0 / self.fps_target
            last_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Display frame (optional for headless MPU systems)
                if 'DISPLAY' in os.environ or os.name == 'nt':
                    cv2.imshow('Crosswalk Detection', processed_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Frame rate control
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_time = time.time()

        except KeyboardInterrupt:
            print("Stopping detection...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.speak_or_print("Detection system stopped")

def main():
    """Main function optimized for MPU deployment."""
    print("Lightweight Crosswalk Detection System")
    print("Optimized for MPU/Embedded Systems")
    print("=" * 40)

    # Try to find model file
    model_files = ['best.pt', 'crosswalk_model.pt', 'yolov8n.pt']
    model_path = None

    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            break

    if model_path:
        print(f"Found model: {model_path}")
    else:
        print("No model found, using basic computer vision detection")

    # Initialize detector
    detector = LightweightCrosswalkDetector(model_path=model_path)

    print("Starting camera... (Press Ctrl+C to stop)")
    detector.run_camera()

if __name__ == "__main__":
    main()