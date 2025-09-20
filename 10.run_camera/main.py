import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import threading
import time
from queue import Queue

class RealTimeTFLiteInference:
    def __init__(self, model_path, input_size=(224, 224), camera_width=1280, camera_height=800):
        """
        Initialize real-time TensorFlow Lite inference with OpenCV

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

        # Initialize camera after model loading
        print("Starting camera initialization...")
        self.cap = None

        # Try V4L2 backend with YUYV format (most compatible)
        print("Trying V4L2 with YUYV format...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if self.cap.isOpened():
            # Set YUYV format which is well supported
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to reduce timeout issues
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("✓ V4L2 with YUYV opened successfully")

        # If YUYV fails, try default backend
        if not self.cap or not self.cap.isOpened():
            print("YUYV V4L2 failed, trying default backend...")
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                print("✓ Default backend opened")

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Cannot open camera with any backend")

        print(f"Camera initialized with backend: {self.cap.getBackendName()}")

        # Test frame read with timeout handling
        print("Testing frame capture...")
        test_success = False
        for i in range(3):
            ret, test_frame = self.cap.read()
            if ret:
                print(f"✓ Frame capture test successful (attempt {i+1})")
                test_success = True
                break
            else:
                print(f"✗ Frame capture test failed (attempt {i+1})")
                time.sleep(0.5)

        if not test_success:
            print("Warning: Frame capture test failed, but proceeding anyway")

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
                    print("Camera connection lost, attempting to reconnect...")
                    time.sleep(1.0)
                    continue

                # Read frame with timeout handling
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    frame_timeout_count += 1
                    consecutive_failures += 1

                    if consecutive_failures > 50:  # If too many consecutive failures
                        print("Too many consecutive frame failures, taking a longer break...")
                        time.sleep(2.0)
                        consecutive_failures = 0
                    else:
                        time.sleep(0.1)

                    if frame_timeout_count % 50 == 0:  # Print every 50 failures
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
                else:
                    # Queue is full, skip this frame to avoid blocking
                    pass

            except Exception as e:
                print(f"Error in capture worker: {e}")
                time.sleep(0.5)  # Longer sleep on exceptions
        
    
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

                print(f"DEBUG: Model output shape: {output_data.shape}, dtype: {output_data.dtype}")

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

        # Convert from NHWC (batch, height, width, channels) to NCHW (batch, channels, height, width)
        input_data = np.transpose(input_data, (0, 3, 1, 2))

        print(f"DEBUG: Preprocessed frame shape: {input_data.shape}, dtype: {input_data.dtype}")
        print(f"DEBUG: Input data range: [{input_data.min():.3f}, {input_data.max():.3f}]")

        return input_data
    
    def postprocess_output(self, output_data):
        """Post-process model output"""
        # Based on output shape [1, 39, 8400], this appears to be a YOLO detection model
        # 39 = 4 (bbox coordinates) + 1 (objectness) + 34 (class probabilities for 34 classes)

        print(f"DEBUG: Processing output with shape: {output_data.shape}")

        if len(output_data.shape) == 3 and output_data.shape[1] == 39:  # YOLO detection format
            detections = output_data[0].T  # Shape: [8400, 39]

            # Extract components
            boxes = detections[:, :4]  # x, y, w, h
            objectness = detections[:, 4]  # objectness score
            class_probs = detections[:, 5:]  # class probabilities

            # Find detections with confidence > threshold
            confidence_threshold = 0.5
            max_class_probs = np.max(class_probs, axis=1)
            confidences = objectness * max_class_probs
            valid_detections = confidences > confidence_threshold

            if np.any(valid_detections):
                valid_boxes = boxes[valid_detections]
                valid_confidences = confidences[valid_detections]
                valid_classes = np.argmax(class_probs[valid_detections], axis=1)

                results = []
                for i in range(len(valid_boxes)):
                    results.append({
                        'bbox': valid_boxes[i].tolist(),
                        'confidence': float(valid_confidences[i]),
                        'class': int(valid_classes[i])
                    })

                return {
                    'type': 'detection',
                    'detections': results,
                    'total_detections': len(results)
                }
            else:
                return {
                    'type': 'detection',
                    'detections': [],
                    'total_detections': 0
                }
        else:
            # Generic output for other model types
            return {
                'type': 'generic',
                'output_shape': output_data.shape,
                'max_value': float(np.max(output_data)),
                'min_value': float(np.min(output_data))
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
        camera_width=1280,
        camera_height=800
    )
    
    inference_system.run()

if __name__ == "__main__":
    main()