#!/bin/sh

# python main_new.py --model  best.onnx
python main_new.py --model best_int8.tflite 
python main.py --model best_int8.tflite --image image.jpg
# 使用極低閾值來看檢測結構
python test.py --model best_int8.tflite --image image.jpg --conf 0.001

# 如果成功檢測，可以逐步提高閾值
python test.py --model best_int8.tflite --image image.jpg --conf 0.1
