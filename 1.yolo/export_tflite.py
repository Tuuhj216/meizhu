from ultralytics import YOLO

# Load your segmentation model
model = YOLO('runs/train/crosswalk_detection6/weights/best.pt')

# Export to TFLite with INT8 quantization
model.export(format='tflite', int8=True, data='./dataset_k_fold/unified_data.yaml', imgsz=640, fraction=0.1)
