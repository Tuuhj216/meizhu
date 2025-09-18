import cv2
import numpy as np
import pyttsx3
import threading
from ultralytics import YOLO
from typing import Tuple, Optional
import time

class CrosswalkDetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """Initialize crosswalk detection system for blind assistance."""
        self.model = YOLO(model_path)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 1.0)
        self.last_voice_time = 0
        self.voice_cooldown = 3  # seconds between voice messages

    def speak(self, text: str):
        """Text-to-speech with cooldown to prevent spam."""
        current_time = time.time()
        if current_time - self.last_voice_time > self.voice_cooldown:
            self.last_voice_time = current_time
            threading.Thread(target=lambda: self.tts_engine.say(text) or self.tts_engine.runAndWait(), daemon=True).start()

    def detect_crosswalk_lines(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Detect crosswalk lines and determine alignment direction.
        Returns: (has_crosswalk, direction_guidance)
        """
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=50, maxLineGap=10)

        if lines is None:
            return False, None

        # Filter for horizontal lines (crosswalk stripes)
        horizontal_lines = []
        height, width = frame.shape[:2]

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Keep lines that are roughly horizontal (within 20 degrees)
            if angle < 20 or angle > 160:
                horizontal_lines.append(line[0])

        if len(horizontal_lines) < 3:  # Need multiple parallel lines for crosswalk
            return False, None

        # Calculate center of detected lines
        line_centers = []
        for x1, y1, x2, y2 in horizontal_lines:
            center_x = (x1 + x2) / 2
            line_centers.append(center_x)

        avg_line_center = np.mean(line_centers)
        frame_center = width / 2

        # Determine alignment guidance
        threshold = width * 0.1  # 10% of frame width

        if avg_line_center < frame_center - threshold:
            return True, "Move right to align with crosswalk"
        elif avg_line_center > frame_center + threshold:
            return True, "Move left to align with crosswalk"
        else:
            return True, "Crosswalk detected, you are aligned"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for crosswalk detection and provide guidance."""
        has_crosswalk, guidance = self.detect_crosswalk_lines(frame)

        # Draw detection results on frame
        if has_crosswalk:
            cv2.putText(frame, "CROSSWALK DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if guidance:
                cv2.putText(frame, guidance, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                self.speak(guidance)
        else:
            cv2.putText(frame, "No crosswalk detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def run_camera(self, camera_index: int = 0):
        """Run real-time crosswalk detection from camera."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        self.speak("Crosswalk detection system started")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Display frame
                cv2.imshow('Crosswalk Detection', processed_frame)

                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Stopping detection...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.speak("Detection system stopped")

def main():
    """Main function to run the crosswalk detection system."""
    detector = CrosswalkDetector()

    print("Crosswalk Detection System for Blind Assistance")
    print("Press 'q' to quit the application")
    print("Starting camera...")

    detector.run_camera()

if __name__ == "__main__":
    main()