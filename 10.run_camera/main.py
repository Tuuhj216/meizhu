import numpy as np
#import tflite_runtime.interpreter as tflite
from tensorflow.lite.python.interpreter import Interpreter
import cv2
import threading
import time
from queue import Queue

class RealTimeTFLiteInference:
    def __init__(self, model_path,):
        """
        Initialize real-time TensorFlow Lite inference with OpenCV

        Args:
            model_path (str): Path to .tflite model
            input_size (tuple): Input size for the model (width, height)
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        # Load TensorFlow Lite model
        #self.interpreter = tflite.Interpreter(model_path=model_path)  #for linux
        self.interpreter = Interpreter(model_path=model_path) #for win
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # ✅ 自動抓模型的輸入大小
        self.input_size = tuple(self.input_details[0]['shape'][1:3])

        # Frame queue for processing
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.running = False
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Expected input size: {self.input_size}")
        
    def capture_worker(self):
        """Worker thread for capturing frames from camera"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
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
                time.sleep(0.1)
        
    
    def inference_worker(self):
        """Worker thread for running TensorFlow Lite inference"""
        while True:
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
                continue
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        """Preprocess frame for model input"""
        input_shape = self.input_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']

        # Resize to model input size
        frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))

        # Convert to RGB if needed
        if frame_resized.shape[2] == 3 and input_dtype == np.uint8:
            input_data = frame_resized.astype(np.uint8)
        elif input_dtype == np.float32:
            input_data = frame_resized.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype}")

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    
    def postprocess_output(self, output_data):
        """Post-process model output"""
        # This depends on your model's output format
        # For classification: return class probabilities
        # For detection: return bounding boxes and classes
        # For segmentation: return segmentation mask
        
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
        while True:
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
        model_path=args.model
        #input_size=(args.width, args.height)
    )
    
    inference_system.run()

if __name__ == "__main__":
    main()