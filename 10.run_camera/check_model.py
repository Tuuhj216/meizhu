#!/usr/bin/env python3

import tflite_runtime.interpreter as tflite
import numpy as np

def check_tflite_model(model_path):
    """Check TFLite model input/output details"""
    print(f"Analyzing model: {model_path}")

    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("\n=== INPUT DETAILS ===")
        for i, detail in enumerate(input_details):
            print(f"Input {i}:")
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Dtype: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")

        print("\n=== OUTPUT DETAILS ===")
        for i, detail in enumerate(output_details):
            print(f"Output {i}:")
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Dtype: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")

        # Test with dummy input
        print("\n=== TESTING DUMMY INPUT ===")
        input_shape = input_details[0]['shape']
        if input_shape[0] == -1:  # Dynamic batch size
            input_shape = [1] + list(input_shape[1:])

        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        print(f"Creating dummy input with shape: {dummy_input.shape}")

        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output.flatten()[:5]}")

    except Exception as e:
        print(f"Error analyzing model: {e}")

if __name__ == "__main__":
    check_tflite_model("best_converted.tflite")