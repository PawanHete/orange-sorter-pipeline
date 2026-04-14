"""
==========================================================
 ORANGE QUALITY DETECTION SYSTEM — CONFIGURATION
==========================================================
 Camera : OAK-D Pro W (RVC2 / Myriad X)
 Host   : Raspberry Pi 4B (8GB)
 Models : YOLO11n (.blob) + ResNet Classifier (.blob)
==========================================================
"""

import os

# ----------------------------------------------------------
# MODEL PATHS (relative to project root)
# ----------------------------------------------------------
# Base directory — assumes script is run from project root
# On Pi: ~/orange_sorter/
# On Windows: E:\Orange POC\
# BASE_DIR evaluates perfectly to the root directory (orange-sorter-pipeline)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stage 1: YOLO11n Orange Detector (single-class)
# Input: [1, 3, 640, 640] — Output: [1, 5, 8400]
# 5 = 4 bbox (cx, cy, w, h) + 1 class score
DETECTION_BLOB = os.path.join(BASE_DIR, "models", "Orange_yolo11n_stereocamera_model.blob")

# Stage 2: ResNet Health Classifier (two-class)
# Input: [1, 3, 224, 224] — Output: [1, 2] (raw logits, needs softmax)
# Index 0 = Healthy, Index 1 = Unhealthy
CLASSIFIER_BLOB = os.path.join(BASE_DIR, "models", "orange_classification_stereocamera.blob")

# ----------------------------------------------------------
# YOLO MODEL CONFIGURATION
# ----------------------------------------------------------
YOLO_INPUT_SIZE = 640                # Model expects 640x640
YOLO_NUM_CLASSES = 1                 # Single class: "orange"
YOLO_CONFIDENCE_THRESHOLD = 0.5     # Minimum detection confidence
YOLO_IOU_THRESHOLD = 0.5            # NMS IoU threshold
YOLO_ANCHORS = []                    # YOLO11 is anchor-free

# ----------------------------------------------------------
# CLASSIFIER CONFIGURATION
# ----------------------------------------------------------
CLASSIFIER_INPUT_SIZE = 224          # Model expects 224x224
CLASSIFIER_THRESHOLD = 0.5          # Softmax prob > 0.5 → Unhealthy
# Output indices after softmax
CLASS_HEALTHY_IDX = 0
CLASS_UNHEALTHY_IDX = 1

# ----------------------------------------------------------
# CAMERA SETTINGS — OAK-D Pro W
# ----------------------------------------------------------
# RGB Camera (IMX378, 12MP)
RGB_FPS = 30
PREVIEW_WIDTH = 640                  # Preview stream to host
PREVIEW_HEIGHT = 480

# Mono Cameras (OV9282, 1MP) — for stereo depth
MONO_FPS = 30

# ----------------------------------------------------------
# STEREO DEPTH SETTINGS
# ----------------------------------------------------------
STEREO_EXTENDED_DISPARITY = True     # Enables closer MinZ (~20cm at 400P)
STEREO_SUBPIXEL = True               # Better depth precision
STEREO_LR_CHECK = True               # Left-Right consistency check
DEPTH_ALIGN_TO_RGB = True            # Align depth map to RGB frame

# Depth ROI median calculation — how much of the bbox to sample
# 0.0 = full bbox, negative = shrink inward (sample center area)
DEPTH_ROI_SHRINK_FACTOR = 0.2        # Sample center 60% of bbox for depth

# ----------------------------------------------------------
# IR PROJECTOR SETTINGS (OAK-D Pro W Active Stereo)
# ----------------------------------------------------------
# The IR dot projector helps with texture-less surfaces (like oranges)
IR_DOT_PROJECTOR_INTENSITY = 0.5     # 0.0 to 1.0 — 50% power
IR_FLOOD_LIGHT_INTENSITY = 0.0       # 0.0 = off (set >0 for dark environments)

# ----------------------------------------------------------
# SIZE GRADING (diameter in mm)
# ----------------------------------------------------------
SIZE_SMALL_MAX = 60                  # < 60mm = Small
SIZE_MEDIUM_MAX = 75                 # 60-75mm = Medium
SIZE_LARGE_MIN = 75                  # > 75mm = Large

# ----------------------------------------------------------
# DISPLAY SETTINGS
# ----------------------------------------------------------
WINDOW_NAME = "Orange Quality Inspector"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Colors (BGR format for OpenCV)
COLOR_HEALTHY = (0, 255, 0)          # Green
COLOR_UNHEALTHY = (0, 0, 255)        # Red
COLOR_NO_DETECTION = (128, 128, 128) # Gray
COLOR_BBOX_ROI = (255, 255, 0)       # Cyan — camera ROI
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_SIZE_TEXT = (255, 200, 0)      # Light blue — size info

# ----------------------------------------------------------
# LOGGING
# ----------------------------------------------------------
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_ENABLED = True
