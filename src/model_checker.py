import cv2
import numpy as np
import time

# --- UNIVERSAL IMPORT BLOCK ---
# Tries to use the lightweight runtime (Pi), falls back to full TF (Laptop)
try:
    from tflite_runtime.interpreter import Interpreter
    print("✅ Logic: Using tflite-runtime (Raspberry Pi Mode)")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("✅ Logic: Using full TensorFlow (Laptop Mode)")

MODEL_PATH = r'E:\Orange POC\models\optimized_best_full_integer_quant.tflite' # Update if your name is different

try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get Output Details
    output_details = interpreter.get_output_details()[0]
    shape = output_details['shape']
    
    print("\n🔍 --- MODEL INSPECTION REPORT ---")
    print(f"📂 Model File:    {MODEL_PATH}")
    print(f"📐 Output Shape:  {shape}")
    
    # LOGIC TO CHECK MODEL TYPE
    # Shape is usually [1, Channels, Anchors] OR [1, Anchors, Channels]
    
    # Find the small number (Channels)
    if shape[1] < shape[2]:
        channels = shape[1] # e.g., 6 or 84
    else:
        channels = shape[2]
        
    print(f"📊 Channels Found: {channels}")
    
    if channels == 6:
        print("✅ SUCCESS: This is a CUSTOM model (2 Classes + 4 Coords).")
        print("   Expected IDs: 0 (Healthy), 1 (Unhealthy)")
    elif channels == 84:
        print("⚠️ WARNING: This is the DEFAULT YOLO model (80 Classes + 4 Coords).")
        print("   You are using the wrong file! This model detects 'Person', 'Car', etc.")
    else:
        print(f"ℹ️ Custom Model with {channels - 4} classes detected.")

except Exception as e:
    print(f"❌ Error: {e}")