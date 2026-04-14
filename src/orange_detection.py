import cv2
import time
import torch
from ultralytics import YOLO

# --- CONFIGURATION ---
FRUIT_DETECTOR = r'E:\Orange POC\yolov8n.pt'          # Small, fast model to find fruit
DISEASE_MODEL = r'E:\Orange POC\sunday_best.pt'       # Your trained large/medium model
CONFIDENCE = 0.5                # Confidence threshold

# ⚡ GPU SETUP
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if DEVICE == 0 else "CPU"
print(f"🚀 Powered by: {device_name}")

def main():
    print("⏳ Loading Models onto GPU...")
    # Load models directly to the GPU
    gatekeeper = YOLO(FRUIT_DETECTOR)
    doctor = YOLO(DISEASE_MODEL)
    
    # 🌡️ WARMUP ROUTINE
    # The first time GPU runs a model, it takes 2-3 seconds to compile. 
    # We do a 'fake' run here so your camera feed starts smoothly.
    print("🔥 Warming up the GPU (this takes 2 seconds)...")
    dummy_frame = torch.zeros((1, 3, 640, 640), device=DEVICE) # Fake image
    gatekeeper.predict(dummy_frame, verbose=False)
    doctor.predict(dummy_frame, verbose=False)
    print("✅ System Ready!")

    # --- CAMERA SETUP ---
    # Use 0 for Laptop Webcam, or 'http://192.168.x.x:4747/video' for Phone
    URL = 2
    cap = cv2.VideoCapture(URL)
    
    # Set Camera Resolution (If using USB Webcam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("❌ Error reading camera.")
            break

        # Resize for consistent speed (YOLO likes 640px)
        # You can try removing this line if your RTX 3050 is fast enough!
        frame_resized = cv2.resize(frame, (640, 480))

        # -------------------------------------------------
        # 1. FIND THE ORANGE (Gatekeeper)
        # -------------------------------------------------
        # Notice device=DEVICE argument -> Forces GPU usage
        results = gatekeeper.predict(frame_resized, classes=[49], conf=CONFIDENCE, device=DEVICE, verbose=False)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the orange
            h, w, _ = frame_resized.shape
            crop = frame_resized[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            if crop.size > 0:
                # ---------------------------------------------
                # 2. CHECK HEALTH (Doctor)
                # ---------------------------------------------
                # Run the Doctor model on GPU
                d_results = doctor.predict(crop, conf=CONFIDENCE, device=DEVICE, verbose=False)
                
                label = "Healthy"
                color = (0, 255, 0) # Green

                if len(d_results[0].boxes) > 0:
                    top_box = d_results[0].boxes[0]
                    # Check class names
                    class_name = doctor.names[int(top_box.cls[0])]
                    conf = float(top_box.conf[0])
                    
                    # Logic: If 'unhealthy' is in the name, flag it Red
                    if "unhealthy" in class_name.lower():
                        label = f"Unhealthy {conf:.0%}"
                        color = (0, 0, 255) # Red
                    else:
                        label = f"Healthy {conf:.0%}"

                # Draw bounding box
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_resized, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS Counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame_resized, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show Output
        cv2.imshow("RTX 3050 - Orange Detector", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()