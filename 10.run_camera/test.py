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
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

        if not self.cap.isOpened():
            print("Warning: Cannot open camera (OK for image-only mode)")
        
        # Load TensorFlow Lite model
        #self.interpreter = tflite.Interpreter(model_path=model_path)  #for linux
        self.interpreter = Interpreter(model_path=model_path) #for win
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 自動抓模型的輸入大小
        self.input_size = tuple(self.input_details[0]['shape'][1:3])

        # Frame queue for processing
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.running = False

        self.num_classes = 3
        self.class_names = ['left', 'right', 'straight']
        self.conf_thres = 0.1  # 降低閾值用於測試
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Expected input size: {self.input_size}")
        
        # Print output details
        print("=== OUTPUT DETAILS ===")
        for i, od in enumerate(self.output_details):
            print(f"Output {i}: index={od['index']}, shape={od['shape']}, dtype={od['dtype']}")
        print("======================")
        
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
        input_shape = self.input_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']

        # Resize to model input size
        frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))

        # Handle different input data types
        if input_dtype == np.uint8:
            input_data = frame_resized.astype(np.uint8)
        elif input_dtype == np.float32:
            input_data = frame_resized.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype}")

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        return input_data
    
    def get_quantization_params(self, output_index):
        """Get quantization parameters for dequantization"""
        output_detail = self.output_details[output_index]
        quant_params = output_detail.get('quantization_parameters', {})
        
        scales = quant_params.get('scales', [1.0])
        zero_points = quant_params.get('zero_points', [0])
        
        # Handle different formats of scales and zero_points
        if isinstance(scales, np.ndarray):
            scale = float(scales.flatten()[0]) if scales.size > 0 else 1.0
        elif isinstance(scales, (list, tuple)) and len(scales) > 0:
            scale = float(scales[0])
        else:
            scale = float(scales) if not isinstance(scales, (list, tuple)) else 1.0
            
        if isinstance(zero_points, np.ndarray):
            zero_point = int(zero_points.flatten()[0]) if zero_points.size > 0 else 0
        elif isinstance(zero_points, (list, tuple)) and len(zero_points) > 0:
            zero_point = int(zero_points[0])
        else:
            zero_point = int(zero_points) if not isinstance(zero_points, (list, tuple)) else 0
            
        return scale, zero_point
        
    def dequantize(self, tensor, output_index):
        """Dequantize tensor using quantization parameters"""
        scale, zero_point = self.get_quantization_params(output_index)
        
        # 只有當需要反量化時才進行
        if scale != 1.0 or zero_point != 0:
            return scale * (tensor.astype(np.float32) - zero_point)
        else:
            return tensor.astype(np.float32)
    
    def postprocess_output(self):
        try:
           # Step 1: 取得原始 proto tensor（shape: [160, 160, 32]）
            proto = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            # Step 2: 做反量化（如果需要的話）
            proto = self.dequantize(proto, 0)  # 使用 output_index = 0

            # ✅ Step 3: 轉置成 [32, 160, 160]
            proto = proto.transpose(2, 0, 1)

            # Output 1 是 detection head → [1, 39, 8400]
            det_raw = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            det_raw = self.dequantize(det_raw, 1)  # 使用 output_index = 1
            det = det_raw.transpose(1, 0)  # shape: [8400, 39]


            num_classes = len(self.class_names)
            boxes = det[:, :4]
            obj_conf = det[:, 4]
            class_probs = det[:, 5:5 + num_classes]
            mask_coeffs = det[:, -32:]

            # 計算最終置信度與類別
            class_ids = np.argmax(class_probs, axis=1)  # shape: [8400]
            class_conf = np.max(class_probs, axis=1)    # shape: [8400]
            final_conf = obj_conf * class_conf

            print(f"det_raw shape: {det_raw.shape}")
            print(f"det shape after transpose: {det.shape}")
            print(f"proto shape: {proto.shape}")
            

            # 過濾低置信度
            valid_idx = np.where(final_conf > self.conf_thres)[0]
            if len(valid_idx) == 0:
                return {
                    'type': 'no_detection',
                    'message': f'No detections above threshold {self.conf_thres}',
                    'debug_info': {
                        'max_conf': float(np.max(final_conf)),
                        'suggestion': f'Try lowering threshold to {float(np.max(final_conf) / 2):.4f}'
                    }
                }

            # 選擇前幾個高置信度的 detection
            selected = valid_idx[:5]
            directions = []
            

            for i in selected:
                print(f"class_ids[i]: {class_ids[i]}, type: {type(class_ids[i])}")
                class_id = int(class_ids[i])
                direction = self.class_names[class_id]
                confidence = float(final_conf[i])
                bbox = boxes[i].tolist()
                coeff = mask_coeffs[i]  # [32]

                # Step 1: flatten proto → [32, 160*160]
                proto_flat = proto.reshape(32, -1)

                # Step 2: dot product → [160*160]
                mask = np.dot(coeff, proto_flat)

                # Step 3: clip to prevent overflow
                mask = np.clip(mask, -30, 30)

                # Step 4: sigmoid activation
                mask = 1 / (1 + np.exp(-mask))

                # Step 5: reshape to image size
                mask = mask.reshape(160, 160)

                # Step 6: binarize
                mask = (mask > 0.5).astype(np.uint8)

                directions.append({
                    'direction': direction,
                    'confidence': confidence,
                    'class_id': class_id,
                    'bbox': bbox,
                    'mask': mask.tolist()  # 可視化時再轉回 numpy
                })

            # 回傳最高置信度的結果
            best = max(directions, key=lambda x: x['confidence'])
            return {
                'type': 'direction_classification',
                'direction': best['direction'],
                'confidence': best['confidence'],
                'class_id': best['class_id'],
                'bbox': best['bbox'],
                'mask': best['mask'],
                'all_directions': directions
            }

        except Exception as e:
            import traceback
            return {
                'type': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
        
    def results_worker(self):
        """Worker thread for handling inference results"""
        while True:
            try:
                result = self.result_queue.get(timeout=1.0)
                
                # Print results
                print(f"Frame {self.frame_count}:")
                print(f"  Inference time: {result['inference_time']:.3f}s")
                try:
                    res = result['results']
                    if isinstance(res, dict):
                        if res['type'] == 'direction_classification':
                            print(f"  ✓ Direction: {res['direction']}")
                            print(f"  ✓ Confidence: {res['confidence']:.3f}")
                            if 'confidence_info' in res:
                                info = res['confidence_info']
                                print(f"  Details: obj={info['obj_conf']:.3f}, class={info['class_conf']:.3f}")
                        elif res['type'] == 'no_detection':
                            print(f"  ❌ {res['message']}")
                            if 'debug_info' in res:
                                debug = res['debug_info']
                                print(f"  Max obj conf: {debug['max_obj_confidence']:.4f}")
                                print(f"  Max class conf: {debug['max_class_confidence']:.4f}")
                        elif res['type'] == 'error':
                            print(f"  🔥 Error: {res['message']}")
                    else:
                        print(f"  Raw result: {str(res)[:300]}...")
                except Exception as e:
                    print(f"  ⚠️ Failed to print result: {e}")
                
                # Calculate FPS
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(f"  Average FPS: {fps:.2f}")
                
            except Exception as e:
                if "timed out" not in str(e):
                    print(f"Error in results worker: {e}")
                continue
    
    def run_image_file(self, image_path):
        """Run inference on a single image file"""
        print(f"🖼️ Loading image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return

        print(f"Original image shape: {image.shape}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.input_size)
        print(f"Resized image shape: {image_resized.shape}")
        
        # Preprocess
        input_data = self.preprocess_frame(image_resized)
        print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")

        # Run inference
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = time.time() - start_time

        # Get result
        result = self.postprocess_output()

        print(f"\n🔍 Inference completed in {inference_time:.3f}s")
        print("📊 Result:")
        
        if result['type'] == 'direction_classification':
            print(f"  ✅ Direction: {result['direction']}")
            print(f"  ✅ Confidence: {result['confidence']:.3f}")
            if 'confidence_info' in result:
                info = result['confidence_info']
                print(f"  📋 Method: {info.get('method', 'unknown')}")
                if 'obj_conf' in info and 'class_conf' in info:
                    print(f"  📋 Details: obj_conf={info['obj_conf']:.3f}, class_conf={info['class_conf']:.3f}")
                elif 'max_val' in info:
                    print(f"  📋 Max value: {info['max_val']:.3f} at position {info.get('max_idx', 'unknown')}")
                if 'class_probs' in info:
                    print(f"  📋 Class probabilities: {[f'{p:.3f}' for p in info['class_probs']]}")
            
                if 'all_directions' in result and len(result['all_directions']) > 1:
                    other_detections = []
                    for d in result['all_directions'][1:3]:
                        other_detections.append((d['direction'], f"{d['confidence']:.3f}"))
                    print(f"  📋 Other detections: {other_detections}")

        elif result['type'] == 'no_detection':
            print(f"  ❌ {result['message']}")
            if 'debug_info' in result:
                debug = result['debug_info']
                print(f"  📊 Max value found: {debug.get('max_value_found', 0):.4f}")
                print(f"  📊 Current threshold: {debug.get('threshold', 0)}")
                if 'suggestion' in debug:
                    print(f"  💡 Suggestion: {debug['suggestion']}")
        elif result['type'] == 'error':
            print(f"  🔥 Error: {result['message']}")
            if 'traceback' in result:
                print(f"  🔍 Traceback: {result['traceback']}")

    def run_single_frame(self):
        """Capture one frame, run inference, and print result"""
        if not self.cap.isOpened():
            print("❌ Camera not available")
            return
            
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to read frame from camera.")
            return

        print("📷 Captured frame from camera")
        print(f"Frame shape: {frame.shape}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.input_size)
        input_data = self.preprocess_frame(frame_resized)

        # Run inference
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = time.time() - start_time

        result = self.postprocess_output()

        print(f"\n🔍 Inference completed in {inference_time:.3f}s")
        print("📊 Result:")
        print(result)

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
            if self.cap.isOpened():
                self.cap.release()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time TensorFlow Lite inference with camera')
    parser.add_argument('--model', required=True, help='Path to .tflite model')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold')

    args = parser.parse_args()
    
    # Create inference system
    inference_system = RealTimeTFLiteInference(model_path=args.model)
    inference_system.conf_thres = args.conf
    
    if args.image:
        inference_system.run_image_file(args.image)
    else:
        inference_system.run_single_frame()

if __name__ == "__main__":
    main()