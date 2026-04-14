import os
import sys
import time
import threading
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. FIX DISPLAY
os.environ["QT_QPA_PLATFORM"] = "xcb"

# ==========================================
# 🔑 CONFIGURATION
# ==========================================
# PASTE YOUR TOKEN HERE IF YOU WANT TO UPLOAD TO BLYNK
BLYNK_AUTH = 'mYw4T5wtToiJ1sKlkSUIEfV9HRF4DgGW'  # Replace with your Blynk Auth Token
BLYNK_SERVER = 'blr1.blynk.cloud'

ROI_X, ROI_Y = 200, 100
ROI_W, ROI_H = 240, 240
MODEL_PATH = 'model.tflite'
CONFIDENCE_THRESHOLD = 0.60 # Lowered slightly for speed

# ==========================================
# 🚀 CLASS: THREADED CAMERA (THE SPEED BOOSTER)
# ==========================================
class VideoStream:
    """
    Reads frames in a separate thread to prevent the main loop from blocking.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # Low resolution is KEY for Raspberry Pi speed
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # MJPG format is faster to decode
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# 🧠 LOAD MODEL
# ==========================================
print("Loading AI Model...")
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=CONFIDENCE_THRESHOLD,
    max_results=1
)
detector = vision.ObjectDetector.create_from_options(options)

# Start Threaded Camera
print("Starting Threaded Camera...")
cam = VideoStream(src=0).start()
time.sleep(2.0) # Warmup

# FPS Counter
frame_count = 0
start_time = time.time()

print("---------------------------------------")
print("TURBO MODE ACTIVE. Press 'q' to quit.")
print("---------------------------------------")

# ==========================================
# 🔄 MAIN LOOP
# ==========================================
while True:
    # 1. Get Frame (Non-blocking now!)
    full_frame = cam.read()
    if full_frame is None: continue

    # 2. Draw ROI
    cv2.rectangle(full_frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 0), 2)
    
    # 3. Crop for AI
    roi_frame = full_frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    
    # 4. AI Detection (Only runs if crop is valid)
    if roi_frame.size > 0:
        rgb_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
        
        # Heavy task
        detection_result = detector.detect(mp_image)

        # 5. Process Results
        if len(detection_result.detections) > 0:
            detection = detection_result.detections[0]
            bbox = detection.bounding_box
            
            # Map coords back to full screen
            global_x = bbox.origin_x + ROI_X
            global_y = bbox.origin_y + ROI_Y
            
            category = detection.categories[0].category_name
            color = (0, 255, 0) if "healthy" not in category.lower() else (0, 0, 255)
            
            cv2.rectangle(full_frame, (global_x, global_y), 
                         (global_x + bbox.width, global_y + bbox.height), color, 3)
            cv2.putText(full_frame, category, (global_x, global_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 6. Calculate FPS
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    cv2.putText(full_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Turbo Mode', full_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()