#!/usr/bin/env python3

import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import threading
import time
from queue import Queue

class RealTimeTFLiteInference:
    def __init__(self, model_path, input_size=(224, 224), use_simulation=False):
        """
        Initialize real-time TensorFlow Lite inference

        Args:
            model_path (str): Path to .tflite model
            input_size (tuple): Input size for the model (width, height)
            use_simulation (bool): Use simulated camera input instead of real camera
        """
        # Load TensorFlow Lite model first
        print("Loading TensorFlow Lite model...")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = input_size
        self.use_simulation = use_simulation

        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Expected input size: {input_size}")

        if not use_simulation:
            # Try to initialize real camera
            print("Attempting to initialize real camera...")
            self.cap = None

            # Try different approaches
            approaches = [
                ("V4L2", cv2.CAP_V4L2),
                ("Default", cv2.CAP_ANY)
            ]

            for name, backend in approaches:
                try:
                    print(f"Trying {name} backend...")
                    cap = cv2.VideoCapture(0, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 5)  # Very low FPS
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                        # Quick test
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"✓ {name} backend working")
                            self.cap = cap
                            break
                        else:
                            print(f"✗ {name} backend frame test failed")
                            cap.release()
                    else:
                        print(f"✗ {name} backend failed to open")
                except Exception as e:
                    print(f"✗ {name} backend error: {e}")

            if not self.cap:
                print("⚠️  Real camera failed, falling back to simulation mode")
                self.use_simulation = True

        if self.use_simulation:
            print("✓ Using simulation mode (generating test images)")
            self.cap = None

        # Frame queue for processing
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue()

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.running = False

    def generate_test_frame(self):
        """Generate a test frame for simulation"""
        # Create a test pattern with changing colors
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some moving patterns
        offset = (self.frame_count * 10) % 640

        # Color gradient
        for i in range(640):
            color_val = int(255 * abs(np.sin((i + offset) * 0.01)))
            frame[:, i] = [color_val, 128, 255 - color_val]

        # Add some shapes
        center_x = 320 + int(100 * np.sin(self.frame_count * 0.1))
        center_y = 240 + int(50 * np.cos(self.frame_count * 0.1))
        cv2.circle(frame, (center_x, center_y), 50, (255, 255, 255), -1)

        # Add frame counter text
        cv2.putText(frame, f"Frame {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return frame

    def capture_worker(self):
        """Worker thread for capturing frames from camera or simulation"""
        frame_timeout_count = 0

        while self.running:
            try:
                if self.use_simulation:
                    # Generate test frame
                    frame = self.generate_test_frame()
                    time.sleep(1.0 / 15)  # Simulate 15 FPS
                else:
                    # Try to read from real camera
                    if not self.cap or not self.cap.isOpened():
                        print("Camera connection lost")
                        time.sleep(1.0)
                        continue

                    ret, frame = self.cap.read()

                    if not ret or frame is None:
                        frame_timeout_count += 1
                        if frame_timeout_count % 20 == 0:
                            print(f"Failed to read frame from camera (count: {frame_timeout_count})")
                        time.sleep(0.1)
                        continue

                # Reset timeout count on successful frame
                frame_timeout_count = 0

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
                mode = "SIMULATION" if self.use_simulation else "CAMERA"
                print(f"[{mode}] Frame {self.frame_count}:")
                print(f"  Inference time: {result['inference_time']:.3f}s")

                if result['results']['type'] == 'classification':
                    print(f"  Class: {result['results']['class']}")
                    print(f"  Confidence: {result['results']['confidence']:.3f}")
                else:
                    print(f"  Output shape: {result['results']['output_shape']}")

                # Calculate FPS
                if self.frame_count % 10 == 0:
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
            if self.cap:
                self.cap.release()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Real-time TensorFlow Lite inference with camera')
    parser.add_argument('--model', required=True, help='Path to .tflite model')
    parser.add_argument('--width', type=int, default=224, help='Input width')
    parser.add_argument('--height', type=int, default=224, help='Input height')
    parser.add_argument('--simulate', action='store_true', help='Force simulation mode')

    args = parser.parse_args()

    # Create and run inference system
    inference_system = RealTimeTFLiteInference(
        model_path=args.model,
        input_size=(args.width, args.height),
        use_simulation=args.simulate
    )

    inference_system.run()

if __name__ == "__main__":
    main()