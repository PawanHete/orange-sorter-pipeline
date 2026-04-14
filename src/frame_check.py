import cv2
import time

def main():
    print("Starting Camera FPS Test...")
    
    # Try Camera Index 0 first (Change to 1 if this fails)
    CAM_INDEX = 0
    cap = cv2.VideoCapture(CAM_INDEX)

    # Force a resolution (standard for Pi projects)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"Could not open camera {CAM_INDEX}. Trying Index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: No camera found on Index 0 or 1.")
            return

    print(f"Camera opened! Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print("Press 'q' to exit.")

    prev_time = 0
    
    while True:
        # 1. Read Frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # 2. Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 3. Display FPS
        # Green text if FPS > 20, Red if low
        color = (0, 255, 0) if fps > 20 else (0, 0, 255)
        
        cv2.putText(frame, f"Raw FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("FPS Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()