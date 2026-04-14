import cv2
import numpy as np
import time
import sys
import os
from ultralytics import YOLO

# ==========================================
# 🔌 GPIO SETUP (LED Lights)
# ==========================================
# Pin definitions (BCM Numbering)
LED_RED_PIN = 17   # Unhealthy
LED_GREEN_PIN = 27 # Healthy

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(LED_RED_PIN, GPIO.OUT)
    GPIO.setup(LED_GREEN_PIN, GPIO.OUT)
    
    # Reset LEDs on start
    GPIO.output(LED_RED_PIN, GPIO.LOW)
    GPIO.output(LED_GREEN_PIN, GPIO.LOW)
    
    ON_PI = True
    print("✅ GPIO Initialized (Raspberry Pi Mode)")

except ImportError:
    ON_PI = False
    print("⚠️ Windows/PC Detected: GPIO Disabled (Simulation Mode)")
    # Dummy functions to prevent crashing on Windows
    class GPIO_Mock:
        LOW = 0
        HIGH = 1
        def output(self, pin, state): pass
    GPIO = GPIO_Mock()

# Helper to control LEDs safely
def set_leds(red_on, green_on):
    if ON_PI:
        GPIO.output(LED_RED_PIN, GPIO.HIGH if red_on else GPIO.LOW)
        GPIO.output(LED_GREEN_PIN, GPIO.HIGH if green_on else GPIO.LOW)
    else:
        # Visual debug for Windows
        status = []
        if red_on: status.append("🔴 RED ON")
        if green_on: status.append("🟢 GREEN ON")
        if not status: status.append("⚪ ALL OFF")
        # We print this inside the frame loop instead of console to avoid spam

# ==========================================
# 🔧 PLATFORM COMPATIBILITY
# ==========================================
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("❌ CRITICAL: No TensorFlow Lite library found.")
        sys.exit(1)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Use the INT8 model for Pi (or change to 'yolo11n.pt' for Windows)
YOLO_MODEL_NAME = r'E:\Orange POC\src\yolo11n.pt' # Change to 'yolo11n.pt' if testing on PC
CLASSIFIER_MODEL_NAME = r'E:\Orange POC\models\orange_health_int8.tflite'
CLASSIFIER_SIZE = 224

# Confidence Settings
YOLO_CONF_THRESH = 0.4
# We use a stricter threshold for "Unhealthy" to avoid false alarms
UNHEALTHY_CONF_THRESH = 0.50 

TARGET_CLASS_IDS = [49] # Orange ID in COCO

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
        if self.is_quantized:
            input_tensor = np.expand_dims(img, axis=0).astype(self.input_details[0]['dtype'])
        else:
            img = img.astype(np.float32)
            img = (img / 127.5) - 1.0
            input_tensor = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        raw_score = float(output[0][0])
        score = raw_score
        if self.is_quantized:
            scale, zero_point = self.output_details[0]['quantization']
            if scale > 0: score = (raw_score - zero_point) * scale
        
        return score # 0.0=Healthy, 1.0=Unhealthy

# ==========================================
# 🚀 MAIN APPLICATION
# ==========================================
def main():
    print("⏳ Loading Models...")
    detector = YOLO(YOLO_MODEL_NAME)
    classifier = OrangeClassifier(CLASSIFIER_MODEL_NAME)
    
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    roi_defined = False
    roi_box = None
    
    # --- 🧠 LATCH LOGIC MEMORY ---
    # Dictionary to store the "Worst Case" status of every orange ID
    # Format: { id: "ROTTEN" or "PENDING" }
    orange_history = {}
    
    # Stats counters
    total_processed = 0
    total_rotten = 0
    total_fresh = 0
    
    prev_time = 0
    
    print("\n ORANGE SORTER WITH LEDS READY")
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
        
        # Flags to control LEDs for *this specific frame*
        frame_has_rotten = False
        frame_has_orange = False
        
        if roi_frame.size > 0:
            # 1. TRACKING
            results = detector.track(
                roi_frame, persist=True, verbose=False, 
                conf=YOLO_CONF_THRESH, tracker="bytetrack.yaml",
                classes=TARGET_CLASS_IDS
            )
            
            cv2.rectangle(display_frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)

            if results and results[0].boxes and results[0].boxes.id is not None:
                frame_has_orange = True
                
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, obj_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    gx1, gy1 = rx + x1, ry + y1
                    gx2, gy2 = rx + x2, ry + y2
                    
                    # 2. CHECK HISTORY (THE LATCH)
                    # If this ID was EVER seen as rotten, it stays rotten forever.
                    is_permanently_rotten = (obj_id in orange_history and orange_history[obj_id] == "ROTTEN")
                    
                    # 3. CLASSIFY (Only if not already condemned)
                    current_status = "FRESH" # Assume fresh unless detected otherwise
                    prob_unhealthy = 0.0
                    
                    if not is_permanently_rotten:
                        orange_crop = frame[gy1:gy2, gx1:gx2]
                        if orange_crop.size > 0:
                            prob_unhealthy = classifier.predict(orange_crop)
                            
                            # THE CRITICAL DECISION:
                            # If we see rot NOW, mark it PERMANENTLY ROTTEN
                            if prob_unhealthy > UNHEALTHY_CONF_THRESH:
                                orange_history[obj_id] = "ROTTEN"
                                is_permanently_rotten = True
                            else:
                                # It looks healthy right now, but we keep monitoring
                                if obj_id not in orange_history:
                                    orange_history[obj_id] = "PENDING"

                    # 4. VISUALIZATION & LED LOGIC
                    if is_permanently_rotten:
                        color = (0, 0, 255) # Red
                        label = "UNHEALTHY (CONFIRMED)"
                        frame_has_rotten = True # Trigger Red LED
                    else:
                        color = (0, 255, 0) # Green
                        label = "HEALTHY (Checking...)"
                        # Note: We don't increment "Fresh" count yet. 
                        # We only count Fresh when it leaves the screen (in a real sorter).
                        # For LEDs, we light Green while it checks.

                    cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), color, 3)
                    cv2.putText(display_frame, f"ID:{obj_id} {label}", (gx1, gy1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- PHASE C: LED HARDWARE CONTROL ---
        # Logic: 
        # 1. If ANY rotten orange is on screen -> RED LED ON (Alert!)
        # 2. If NO rotten oranges, but YES oranges -> GREEN LED ON (Process OK)
        # 3. If NO oranges -> BOTH OFF
        
        if frame_has_rotten:
            set_leds(red_on=True, green_on=False)
            led_status_text = "LED: 🔴 RED"
        elif frame_has_orange:
            set_leds(red_on=False, green_on=True)
            led_status_text = "LED: 🟢 GREEN"
        else:
            set_leds(red_on=False, green_on=False)
            led_status_text = "LED: ⚪ OFF"

        # --- PHASE D: DASHBOARD ---
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        cv2.rectangle(display_frame, (0, 0), (640, 40), (0, 0, 0), -1)
        stats_text = f"FPS:{fps:.1f} | {led_status_text}"
        cv2.putText(display_frame, stats_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display Latch Status at bottom
        cv2.putText(display_frame, "Rotation Logic Active: 1 Defect = Permanent Fail", (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Orange Sorter LED", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): # Reset
            roi_defined = False
            orange_history.clear()

    cap.release()
    cv2.destroyAllWindows()
    
    if ON_PI:
        GPIO.cleanup()
        print("🔌 GPIO Cleaned up.")

if __name__ == "__main__":
    main()