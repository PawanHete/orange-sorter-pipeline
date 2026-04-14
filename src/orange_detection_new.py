import cv2
import numpy as np
import time
import sys
import os
from ultralytics import YOLO

# ==========================================
# 🔧 PLATFORM COMPATIBILITY
# ==========================================
try:
    import tflite_runtime.interpreter as tflite
    print("✅ Detected Raspberry Pi (using tflite_runtime)")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("✅ Detected Windows/PC (using tensorflow.lite)")
    except ImportError:
        print("❌ CRITICAL: No TensorFlow Lite library found.")
        sys.exit(1)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
YOLO_MODEL_NAME = r'E:\Orange POC\src\yolo11n.pt' 
CLASSIFIER_MODEL_NAME = r'E:\Orange POC\models\orange_health_int8.tflite'
CLASSIFIER_SIZE = 224
YOLO_CONF_THRESH = 0.6

# 🍊 COCO ID for Orange is 49
TARGET_CLASS_IDS = [49] 

# ==========================================
# 🧠 CLASS: CLASSIFIER
# ==========================================
class OrangeClassifier:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"❌ Error: Classifier model '{model_path}' not found!")
            sys.exit(1)
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.is_quantized = self.input_details[0]['dtype'] in [np.int8, np.uint8]

    def predict(self, image):
        img = cv2.resize(image, (CLASSIFIER_SIZE, CLASSIFIER_SIZE))
        
        # Preprocess
        if self.is_quantized:
            input_tensor = np.expand_dims(img, axis=0).astype(self.input_details[0]['dtype'])
        else:
            img = img.astype(np.float32)
            img = (img / 127.5) - 1.0
            input_tensor = np.expand_dims(img, axis=0)

        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Fix: Cast to float to prevent Overflow
        raw_score = float(output[0][0])
        score = raw_score

        if self.is_quantized:
            scale, zero_point = self.output_details[0]['quantization']
            if scale > 0: 
                score = (raw_score - zero_point) * scale
        
        return score # 0.0=Healthy, 1.0=Unhealthy

# ==========================================
# 🚀 MAIN APPLICATION
# ==========================================
def main():
    print("⏳ Loading Models...")
    detector = YOLO(YOLO_MODEL_NAME)
    classifier = OrangeClassifier(CLASSIFIER_MODEL_NAME)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    roi_defined = False
    roi_box = None
    
    # Stats
    counted_ids = set()
    count_healthy = 0
    count_unhealthy = 0
    total_oranges = 0
    prev_time = 0
    
    print("\n🍊 ORANGE SORTER READY")
    print("Select ROI with mouse. Press SPACE to confirm.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        current_time = time.time()
        
        # --- PHASE A: ROI SELECTION ---
        if not roi_defined:
            cv2.putText(display_frame, "SELECT ROI & PRESS SPACE", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Orange Sorter", display_frame)
            try:
                roi_box = cv2.selectROI("Orange Sorter", frame, showCrosshair=True)
                if roi_box[2] > 0 and roi_box[3] > 0:
                    roi_defined = True
                    cv2.destroyWindow("Orange Sorter")
            except: break
            continue

        # --- PHASE B: PRODUCTION ---
        rx, ry, rw, rh = roi_box
        roi_frame = frame[ry:ry+rh, rx:rx+rw]
        
        # Default Status (Assume no orange)
        stage1_status = "NO ORANGES DETECTED"
        stage1_color = (0, 0, 255) # Red
        
        if roi_frame.size > 0:
            # 1. YOLO DETECTION (Only Oranges)
            results = detector.track(
                roi_frame, 
                persist=True, 
                verbose=False, 
                conf=YOLO_CONF_THRESH, 
                tracker="bytetrack.yaml",
                classes=TARGET_CLASS_IDS 
            )
            
            # Draw ROI Box
            cv2.rectangle(display_frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)

            # Check if any orange was found
            if results and results[0].boxes and len(results[0].boxes) > 0:
                # ✅ UPDATE STATUS: ORANGE FOUND
                stage1_status = "ORANGE DETECTED"
                stage1_color = (0, 255, 0) # Green

                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                
                # Only try to get IDs if tracking is successful
                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, obj_id in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        
                        # Coordinate Mapping
                        gx1, gy1 = rx + x1, ry + y1
                        gx2, gy2 = rx + x2, ry + y2
                        
                        # Classification
                        orange_crop = frame[gy1:gy2, gx1:gx2]
                        
                        if orange_crop.size > 0:
                            prob_unhealthy = classifier.predict(orange_crop)
                            
                            is_unhealthy = prob_unhealthy > 0.5
                            label_text = f"ROTTEN {prob_unhealthy:.0%}" if is_unhealthy else f"FRESH {(1-prob_unhealthy):.0%}"
                            color = (0, 0, 255) if is_unhealthy else (0, 255, 0)
                            
                            # Counting Logic
                            if obj_id not in counted_ids:
                                counted_ids.add(obj_id)
                                total_oranges += 1
                                if is_unhealthy:
                                    count_unhealthy += 1
                                else:
                                    count_healthy += 1
                                    
                            cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), color, 3)
                            cv2.putText(display_frame, f"ID:{obj_id} {label_text}", (gx1, gy1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- PHASE C: DASHBOARD & STATUS ---
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # 1. Top Dashboard (Stats)
        cv2.rectangle(display_frame, (0, 0), (640, 40), (0, 0, 0), -1)
        stats_text = f"FPS: {fps:.1f} | TOTAL: {total_oranges} | FRESH: {count_healthy} | ROTTEN: {count_unhealthy}"
        cv2.putText(display_frame, stats_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 2. Bottom Status (Stage 1 Detection Feedback)
        # Draw a semi-transparent black box at the bottom for readability
        cv2.rectangle(display_frame, (0, 440), (640, 480), (0, 0, 0), -1)
        cv2.putText(display_frame, f"STATUS: {stage1_status}", (20, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, stage1_color, 2)

        cv2.imshow("Orange Sorter", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): # Reset
            roi_defined = False
            counted_ids.clear()
            count_healthy = 0
            count_unhealthy = 0
            total_oranges = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()