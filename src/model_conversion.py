from ultralytics import YOLO

# Load your custom trained model
model = YOLO('E:\\Orange POC\\sunday_best.pt')  # Replace with your actual filename

# Export the model to TFLite format
# 'int8=True' is CRITICAL. It converts 32-bit math to 8-bit, 
# making the model 4x smaller and much faster on Pi.
model.export(format='tflite', int8=True)