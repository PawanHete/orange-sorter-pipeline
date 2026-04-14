import cv2
import numpy as np
import time

# --- UNIVERSAL IMPORT BLOCK ---
try:
    from tflite_runtime.interpreter import Interpreter
    print("✅ Hardware: Raspberry Pi (using tflite-runtime)")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("✅ Hardware: Laptop (using full TensorFlow)")

# --- CONFIGURATION ---
MODEL_PATH = r'E:\Orange POC\models\optimized_best_full_integer_quant.tflite'
CONF_THRESHOLD = 0.50
# We will define labels, but we will also add a safety check
LABELS = ['Healthy', 'Unhealthy']

def main():
    print("🚀 Loading AI Model...")
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    model_h = input_details[0]['shape'][1]
    model_w = input_details[0]['shape'][2]
    
    # Check if model expects Integers (INT8)
    input_type = input_details[0]['dtype']
    is_int8 = (input_type == np.int8)
    
    if is_int8:
        scale, zero_point = input_details[0]['quantization']
        scale = float(scale)
        zero_point = int(zero_point)

    # --- CAMERA SETUP ---
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"✅ Camera: 640x480 | Model Input: {model_w}x{model_h}")
    print("Press 'q' to exit.")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- 1. PREPROCESS ---
        img_small = cv2.resize(frame, (model_w, model_h))
        
        if is_int8:
            input_data = (np.float32(img_small) / 255.0) / scale + zero_point
            input_data = input_data.astype(np.int8)
        else:
            input_data = (np.float32(img_small) / 255.0)
            
        input_data = np.expand_dims(input_data, axis=0)

        # --- 2. INFERENCE ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # --- 3. PROCESS RESULTS ---
        # Format: [x, y, w, h, score, class_id]
        for det in output:
            score = det[4]
            
            if score < CONF_THRESHOLD:
                continue 

            class_id = int(det[5])
            
            # Extract Box
            x_center, y_center, w, h = det[0], det[1], det[2], det[3]
            x = x_center - (w / 2)
            y = y_center - (h / 2)
            
            # Scale to Display
            scale_x = 640 / model_w
            scale_y = 480 / model_h
            
            x_final = int(x * scale_x)
            y_final = int(y * scale_y)
            w_final = int(w * scale_x)
            h_final = int(h * scale_y)

            # --- SAFE LABEL LOGIC (Prevents Crash) ---
            if 0 <= class_id < len(LABELS):
                # Normal Case: ID is 0 or 1
                label_text = LABELS[class_id]
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255) # Green vs Red
            else:
                # Error Case: ID is weird (e.g., 2, 49, etc.)
                print(f"⚠️ Warning: Model detected unknown Class ID: {class_id}")
                label_text = f"ID {class_id}"
                color = (255, 255, 0) # Yellow for unknown

            label = f"{label_text} {int(score*100)}%"

            # Draw
            cv2.rectangle(frame, (x_final, y_final), (x_final+w_final, y_final+h_final), color, 2)
            cv2.putText(frame, label, (x_final, y_final-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- FPS Counter ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow("Orange Detector", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()