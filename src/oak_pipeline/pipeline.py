"""
==========================================================
 DEPTHAI PIPELINE BUILDER — OAK-D Pro W (RVC2)
==========================================================
 Builds a two-stage inference pipeline:
   Stage 1: YOLO11n detection (orange detector)
   Stage 2: ResNet classifier (healthy/unhealthy)
   + StereoDepth for size measurement
==========================================================
"""

import depthai as dai
from . import config


def build_pipeline():
    """
    Builds and returns a DepthAI pipeline configured for the OAK-D Pro W.

    Pipeline Nodes:
    ───────────────
    1. ColorCamera (CAM_A)         → RGB stream (preview + NN input)
    2. MonoCamera LEFT (CAM_B)     → Left stereo input
    3. MonoCamera RIGHT (CAM_C)    → Right stereo input
    4. StereoDepth                 → Depth map (aligned to RGB)
    5. NeuralNetwork (Stage 1)     → YOLO11n orange detection (.blob)
    6. ImageManip                  → Crop + resize detected region
    7. NeuralNetwork (Stage 2)     → Health classifier (.blob)
    8. XLinkOut × 4                → preview, nn_out, class_out, depth
    9. XLinkIn × 1                 → manip_cfg (host → device ROI)

    Data Flow:
    ──────────
    RGB Camera ──→ YOLO NN (Stage 1) ──→ [nn_out] ──→ Host
         │                                              │
         │          Host sends ROI ◄────────────────────┘
         │              │
         └──────→ ImageManip (crop/resize 224×224)
                        │
                        └──→ Classifier NN (Stage 2) ──→ [class_out] ──→ Host

    Left Mono ─┐
               ├──→ StereoDepth (aligned RGB) ──→ [depth] ──→ Host
    Right Mono ┘

    Returns:
        dai.Pipeline: The configured pipeline ready to be started
    """
    pipeline = dai.Pipeline()

    # ===========================================================
    # 1. COLOR CAMERA (RGB)
    # ===========================================================
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(config.RGB_FPS)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Preview stream — lower resolution for display on host
    cam_rgb.setPreviewSize(config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT)
    cam_rgb.setPreviewKeepAspectRatio(True)

    # Video stream — used for ImageManip cropping (full res not needed,
    # but we use the ISP output which is already processed)
    # We set the video size to be reasonable for the classifier pipeline
    cam_rgb.setVideoSize(config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT)

    # ===========================================================
    # 2. MONO CAMERAS (Stereo Pair)
    # ===========================================================
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setFps(config.MONO_FPS)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setFps(config.MONO_FPS)

    # ===========================================================
    # 3. STEREO DEPTH
    # ===========================================================
    stereo = pipeline.create(dai.node.StereoDepth)

    stereo.setLeftRightCheck(config.STEREO_LR_CHECK)
    stereo.setExtendedDisparity(config.STEREO_EXTENDED_DISPARITY)
    stereo.setSubpixel(config.STEREO_SUBPIXEL)

    # Align depth map to the RGB camera so depth pixels correspond to RGB pixels
    if config.DEPTH_ALIGN_TO_RGB:
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # Link mono cameras to stereo
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # ===========================================================
    # 4. STAGE 1: YOLO11n DETECTION NEURAL NETWORK
    # ===========================================================
    # Using NeuralNetwork node (not DetectionNetwork) because YOLO11
    # anchor-free output [1, 5, 8400] requires custom post-processing
    # on the host side.
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(config.DETECTION_BLOB)
    detection_nn.setNumInferenceThreads(2)
    detection_nn.input.setBlocking(False)
    detection_nn.input.setQueueSize(1)

    # The YOLO model expects 640x640 input — use ImageManip to resize
    yolo_manip = pipeline.create(dai.node.ImageManip)
    yolo_manip.initialConfig.setResize(config.YOLO_INPUT_SIZE, config.YOLO_INPUT_SIZE)
    yolo_manip.initialConfig.setKeepAspectRatio(False)
    yolo_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)  # planar BGR

    cam_rgb.preview.link(yolo_manip.inputImage)
    yolo_manip.out.link(detection_nn.input)

    # ===========================================================
    # 5. IMAGE MANIP FOR STAGE 2 (Crop + Resize)
    # ===========================================================
    # This node receives ROI configs from the host (via XLinkIn)
    # and crops the RGB frame at that location, resizing to 224x224
    classifier_manip = pipeline.create(dai.node.ImageManip)
    classifier_manip.initialConfig.setResize(
        config.CLASSIFIER_INPUT_SIZE, config.CLASSIFIER_INPUT_SIZE
    )
    classifier_manip.initialConfig.setKeepAspectRatio(False)
    classifier_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    classifier_manip.setWaitForConfigInput(True)  # Wait for ROI from host
    classifier_manip.inputImage.setQueueSize(1)
    classifier_manip.inputImage.setBlocking(False)

    # Link RGB video stream to the classifier manip (source for cropping)
    cam_rgb.video.link(classifier_manip.inputImage)

    # Host sends crop config via XLinkIn
    manip_cfg_in = pipeline.create(dai.node.XLinkIn)
    manip_cfg_in.setStreamName("manip_cfg")
    manip_cfg_in.out.link(classifier_manip.inputConfig)

    # ===========================================================
    # 6. STAGE 2: HEALTH CLASSIFIER NEURAL NETWORK
    # ===========================================================
    classifier_nn = pipeline.create(dai.node.NeuralNetwork)
    classifier_nn.setBlobPath(config.CLASSIFIER_BLOB)
    classifier_nn.setNumInferenceThreads(1)
    classifier_nn.input.setBlocking(False)
    classifier_nn.input.setQueueSize(1)

    # Link cropped image to classifier
    classifier_manip.out.link(classifier_nn.input)

    # ===========================================================
    # 7. XLINK OUTPUTS (Device → Host)
    # ===========================================================

    # Preview frame (for display)
    xout_preview = pipeline.create(dai.node.XLinkOut)
    xout_preview.setStreamName("preview")
    xout_preview.input.setBlocking(False)
    xout_preview.input.setQueueSize(1)
    cam_rgb.preview.link(xout_preview.input)

    # Stage 1 — YOLO raw output
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn_out")
    xout_nn.input.setBlocking(False)
    xout_nn.input.setQueueSize(1)
    detection_nn.out.link(xout_nn.input)

    # Stage 2 — Classifier raw output
    xout_class = pipeline.create(dai.node.XLinkOut)
    xout_class.setStreamName("class_out")
    xout_class.input.setBlocking(False)
    xout_class.input.setQueueSize(1)
    classifier_nn.out.link(xout_class.input)

    # Depth map
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    xout_depth.input.setBlocking(False)
    xout_depth.input.setQueueSize(1)
    stereo.depth.link(xout_depth.input)

    return pipeline
