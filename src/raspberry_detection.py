import cv2
import torch
from ultralytics import YOLO

# --- CONFIGURATION ---
# Path to your new retrained Nano model (.pt)
MODEL_PATH = r'E:\Orange POC\models\optimized_nano_best.pt' 
CONFIDENCE = 0.5

# ⚡ GPU SETUP
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"🚀 Running on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

def main():
    # Load the single model (No more Gatekeeper/Doctor split)
    print(f"⏳ Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # --- CAMERA SETUP ---
    # Use 0 for Webcam, or video file path
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # -------------------------------------------------
        # SINGLE PASS DETECTION
        # -------------------------------------------------
        # The model now finds the orange AND classifies it in one go.
        # verbose=False keeps the console clean.
        results = model.predict(frame, conf=CONFIDENCE, device=device, verbose=False)

        # Visualize the results on the frame automatically
        # Ultralytics has a built-in plotter that draws boxes and labels nicely
        annotated_frame = results[0].plot()

        # Show Output
        cv2.imshow("Laptop - PT Model Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()