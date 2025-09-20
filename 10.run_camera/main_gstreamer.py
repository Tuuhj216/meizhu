#!/usr/bin/env python3

import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import threading
import time
from queue import Queue

class RealTimeTFLiteInference:
    def __init__(self, model_path, input_size=(224, 224), camera_width=640, camera_height=480):
        """
        Initialize real-time TensorFlow Lite inference with GStreamer

        Args:
            model_path (str): Path to .tflite model
            input_size (tuple): Input size for the model (width, height)
            camera_width (int): Camera capture width
            camera_height (int): Camera capture height
        """
        # Load TensorFlow Lite model first
        print("Loading TensorFlow Lite model...")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = input_size

        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Expected input size: {input_size}")

        # Initialize camera with GStreamer
        print("Starting camera initialization with GStreamer...")

        # GStreamer pipeline for camera capture
        gst_pipeline = (
            f"v4l2src device=/dev/video0 ! "
            f"video/x-raw,format=YUY2,width={camera_width},height={camera_height},framerate=15/1 ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1"
        )

        print(f"GStreamer pipeline: {gst_pipeline}")

        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera with GStreamer pipeline")

        print("✓ Camera initialized with GStreamer")

        # Test frame capture
        print("Testing frame capture...")
        ret, test_frame = self.cap.read()
        if ret and test_frame is not None:
            print(f"✓ Frame capture test successful, frame shape: {test_frame.shape}")
        else:
            print("✗ Frame capture test failed")

        # Frame queue for processing
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue()

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.running = False

    def capture_worker(self):
        """Worker thread for capturing frames from camera"""
        frame_timeout_count = 0
        consecutive_failures = 0

        while self.running:
            try:
                # Check if camera is still opened
                if not self.cap.isOpened():
                    print("Camera connection lost")
                    time.sleep(1.0)
                    continue

                # Read frame
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    frame_timeout_count += 1
                    consecutive_failures += 1

                    if consecutive_failures > 10:  # Fewer retries with GStreamer
                        print("Too many consecutive frame failures, taking a break...")
                        time.sleep(1.0)
                        consecutive_failures = 0
                    else:
                        time.sleep(0.1)

                    if frame_timeout_count % 20 == 0:  # Print every 20 failures
                        print(f"Failed to read frame from camera (count: {frame_timeout_count})")
                    continue

                # Reset failure counts on successful read
                frame_timeout_count = 0
                consecutive_failures = 0

                # Validate frame
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    print("Invalid frame dimensions, skipping...")
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to input size
                frame_resized = cv2.resize(frame_rgb, self.input_size)

                # Add frame to queue for processing (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_resized.copy())

            except Exception as e:
                print(f"Error in capture worker: {e}")
                time.sleep(0.5)

    def inference_worker(self):
        """Worker thread for running TensorFlow Lite inference"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)

                # Preprocess frame
                input_data = self.preprocess_frame(frame)

                # Run inference
                start_time = time.time()
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                inference_time = time.time() - start_time

                # Post-process results
                results = self.postprocess_output(output_data)

                # Add results to output queue
                self.result_queue.put({
                    'results': results,
                    'inference_time': inference_time,
                    'frame_shape': frame.shape
                })

                self.frame_count += 1

            except Exception as e:
                if "timed out" not in str(e):
                    print(f"Error in inference worker: {e}")
                if not self.running:
                    break
                continue

    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Normalize to [0, 1]
        input_data = frame.astype(np.float32) / 255.0

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    def postprocess_output(self, output_data):
        """Post-process model output"""
        if len(output_data.shape) == 2:  # Classification
            probabilities = output_data[0]
            top_class = np.argmax(probabilities)
            confidence = probabilities[top_class]
            return {
                'type': 'classification',
                'class': int(top_class),
                'confidence': float(confidence),
                'probabilities': probabilities.tolist()
            }
        else:
            # Generic output for other model types
            return {
                'type': 'generic',
                'output_shape': output_data.shape,
                'output': output_data.tolist()
            }

    def results_worker(self):
        """Worker thread for handling inference results"""
        while self.running:
            try:
                result = self.result_queue.get(timeout=1.0)

                # Print results
                print(f"Frame {self.frame_count}:")
                print(f"  Inference time: {result['inference_time']:.3f}s")
                print(f"  Results: {result['results']}")

                # Calculate FPS
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(f"  Average FPS: {fps:.2f}")

            except Exception as e:
                if "timed out" not in str(e):
                    print(f"Error in results worker: {e}")
                if not self.running:
                    break
                continue

    def run(self):
        """Start the real-time inference system"""
        print("Starting real-time inference...")

        self.running = True

        # Start worker threads
        capture_thread = threading.Thread(target=self.capture_worker, daemon=True)
        inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        results_thread = threading.Thread(target=self.results_worker, daemon=True)

        capture_thread.start()
        inference_thread.start()
        results_thread.start()

        try:
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            self.cap.release()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Real-time TensorFlow Lite inference with camera')
    parser.add_argument('--model', required=True, help='Path to .tflite model')
    parser.add_argument('--width', type=int, default=224, help='Input width')
    parser.add_argument('--height', type=int, default=224, help='Input height')

    args = parser.parse_args()

    # Create and run inference system
    inference_system = RealTimeTFLiteInference(
        model_path=args.model,
        input_size=(args.width, args.height),
        camera_width=640,
        camera_height=480
    )

    inference_system.run()

if __name__ == "__main__":
    main()