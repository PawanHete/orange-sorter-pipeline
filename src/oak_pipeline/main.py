#!/usr/bin/env python3
"""
==========================================================
 ORANGE QUALITY INSPECTOR — Main Application
==========================================================
 Two-stage pipeline on OAK-D Pro W (RVC2) + Raspberry Pi 4B

 Stage 1: YOLO11n detects oranges (single-class, .blob)
 Stage 2: ResNet classifies healthy/unhealthy (.blob)
 + Stereo depth measures real-world diameter (mm)

 Usage:
   python3 main.py                    # Full pipeline
   python3 main.py --preview-only     # Camera preview only
   python3 main.py --detection-only   # Stage 1 only (no classifier)
   python3 main.py --show-depth       # Show depth map overlay

 Controls:
   q — Quit
   r — Reset counters
   s — Save screenshot
   d — Toggle depth overlay
==========================================================
"""

import sys
import os
import time
import argparse
import csv
from datetime import datetime

import cv2
import numpy as np
import depthai as dai

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oak_pipeline import config
from oak_pipeline.pipeline import build_pipeline
from oak_pipeline.yolo_postprocess import parse_yolo_output
from oak_pipeline.size_calculator import SizeCalculator
from oak_pipeline import display


def softmax(x):
    """Compute softmax over the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def parse_classifier_output(nn_data):
    """
    Parse the classifier NN output.

    The classifier outputs raw logits [1, 2]:
        Index 0 = Healthy score
        Index 1 = Unhealthy score

    Apply softmax to get probabilities.

    Args:
        nn_data: dai.NNData — raw output from the classifier NN

    Returns:
        tuple: (is_healthy: bool, healthy_prob: float, unhealthy_prob: float)
    """
    # Get the first (and only) output layer
    output_layer = nn_data.getFirstLayerFp16()
    logits = np.array(output_layer).reshape(1, 2)

    # Apply softmax
    probs = softmax(logits)[0]

    healthy_prob = float(probs[config.CLASS_HEALTHY_IDX])
    unhealthy_prob = float(probs[config.CLASS_UNHEALTHY_IDX])

    is_healthy = unhealthy_prob < config.CLASSIFIER_THRESHOLD

    return is_healthy, healthy_prob, unhealthy_prob


def setup_logging():
    """Create log directory and CSV file for session results."""
    if not config.LOG_ENABLED:
        return None

    os.makedirs(config.LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(config.LOG_DIR, f"session_{timestamp}.csv")

    f = open(log_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "detection_id", "confidence",
        "verdict", "healthy_prob", "unhealthy_prob",
        "diameter_mm", "grade", "depth_mm"
    ])
    print(f"[Logging] Session log: {log_path}")
    return writer, f


def main():
    # ----------------------------------------------------------
    # ARGUMENT PARSING
    # ----------------------------------------------------------
    parser = argparse.ArgumentParser(description="Orange Quality Inspector")
    parser.add_argument("--preview-only", action="store_true",
                        help="Show camera preview only (no AI)")
    parser.add_argument("--detection-only", action="store_true",
                        help="Run Stage 1 detection only (no classifier)")
    parser.add_argument("--show-depth", action="store_true",
                        help="Show depth map overlay")
    args = parser.parse_args()

    # ----------------------------------------------------------
    # STARTUP
    # ----------------------------------------------------------
    print("=" * 55)
    print("  ORANGE QUALITY INSPECTOR")
    print("  Camera : OAK-D Pro W (RVC2)")
    print("  Host   : Raspberry Pi 4B (8GB)")
    print(f"  DepthAI: {dai.__version__}")
    print("=" * 55)

    # Verify model files exist
    if not args.preview_only:
        if not os.path.exists(config.DETECTION_BLOB):
            print(f"[ERROR] Detection model not found: {config.DETECTION_BLOB}")
            sys.exit(1)
        if not args.detection_only and not os.path.exists(config.CLASSIFIER_BLOB):
            print(f"[ERROR] Classifier model not found: {config.CLASSIFIER_BLOB}")
            sys.exit(1)

    # ----------------------------------------------------------
    # BUILD PIPELINE
    # ----------------------------------------------------------
    if args.preview_only:
        print("\n[Mode] PREVIEW ONLY — no AI models loaded")
        pipeline = _build_preview_pipeline()
    elif args.detection_only:
        print("\n[Mode] DETECTION ONLY — Stage 1 only")
        pipeline = build_pipeline()
    else:
        print("\n[Mode] FULL PIPELINE — Detection + Classification + Size")
        pipeline = build_pipeline()

    # ----------------------------------------------------------
    # START DEVICE
    # ----------------------------------------------------------
    print("\n[Device] Connecting to OAK-D Pro W...")

    with dai.Device(pipeline) as device:
        print(f"[Device] Connected! USB speed: {device.getUsbSpeed().name}")

        # Enable IR dot projector for better depth on smooth surfaces
        device.setIrLaserDotProjectorIntensity(config.IR_DOT_PROJECTOR_INTENSITY)
        device.setIrFloodLightIntensity(config.IR_FLOOD_LIGHT_INTENSITY)
        print(f"[Device] IR Dot Projector: {config.IR_DOT_PROJECTOR_INTENSITY * 100:.0f}%")

        # Initialize size calculator (reads camera calibration)
        size_calc = None
        if not args.preview_only:
            try:
                size_calc = SizeCalculator(device)
            except Exception as e:
                print(f"[WARNING] Size calculator init failed: {e}")
                print("[WARNING] Size measurement will be disabled")

        # Setup output queues
        q_preview = device.getOutputQueue("preview", maxSize=1, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)

        q_nn = None
        q_class = None
        q_manip_cfg = None

        if not args.preview_only:
            q_nn = device.getOutputQueue("nn_out", maxSize=1, blocking=False)

            if not args.detection_only:
                q_class = device.getOutputQueue("class_out", maxSize=1, blocking=False)
                q_manip_cfg = device.getInputQueue("manip_cfg", maxSize=1, blocking=False)

        # Setup logging
        log_writer = None
        log_file = None
        if config.LOG_ENABLED and not args.preview_only:
            try:
                log_writer, log_file = setup_logging()
            except Exception as e:
                print(f"[WARNING] Logging setup failed: {e}")

        # ----------------------------------------------------------
        # STATS
        # ----------------------------------------------------------
        total_oranges = 0
        count_healthy = 0
        count_unhealthy = 0
        detection_counter = 0
        show_depth_overlay = args.show_depth

        # FPS tracking
        fps = 0.0
        frame_count = 0
        fps_start_time = time.monotonic()

        print("\n" + "-" * 55)
        print("  RUNNING — Press 'q' to quit, 'r' to reset, 'd' toggle depth")
        print("-" * 55 + "\n")

        # ----------------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------------
        while True:
            # 1. Get preview frame
            in_preview = q_preview.tryGet()
            if in_preview is None:
                continue

            frame = in_preview.getCvFrame()

            # Get depth frame
            depth_frame = None
            in_depth = q_depth.tryGet()
            if in_depth is not None:
                depth_frame = in_depth.getFrame()  # uint16, values in mm

            # FPS calculation
            frame_count += 1
            elapsed = time.monotonic() - fps_start_time
            if elapsed > 0.5:  # Update FPS every 0.5 seconds
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.monotonic()

            # ----- PREVIEW ONLY MODE -----
            if args.preview_only:
                display.draw_dashboard(frame, fps, 0, 0, 0)
                display.draw_status_bar(frame, "PREVIEW MODE", (0, 255, 255))
                if show_depth_overlay and depth_frame is not None:
                    _overlay_depth(frame, depth_frame)
                cv2.imshow(config.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('d'):
                    show_depth_overlay = not show_depth_overlay
                continue

            # ----- AI PROCESSING -----
            status_text = "NO ORANGES DETECTED"
            status_color = config.COLOR_NO_DETECTION

            # 2. Get Stage 1 (YOLO) output
            in_nn = q_nn.tryGet()
            detections = []

            if in_nn is not None:
                # Get raw output tensor
                raw_output = np.array(in_nn.getFirstLayerFp16()).reshape(1, 5, 8400)

                # Parse YOLO output into bounding boxes
                detections = parse_yolo_output(
                    raw_output, config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT
                )

            if len(detections) > 0:
                status_text = f"ORANGE DETECTED ({len(detections)})"
                status_color = config.COLOR_HEALTHY

                for det in detections:
                    bbox_pixel = det['bbox_pixel']
                    det_conf = det['confidence']
                    bbox_norm = det['bbox']

                    # Default: just detection, no classification
                    label = "ORANGE"
                    color = config.COLOR_BBOX_ROI
                    class_conf = det_conf
                    size_info = None

                    # 3. Stage 2: Classify (if not detection-only mode)
                    if not args.detection_only and q_manip_cfg is not None and q_class is not None:
                        # Send crop ROI to ImageManip on device
                        manip_cfg = dai.ImageManipConfig()
                        manip_cfg.setCropRect(
                            bbox_norm[0], bbox_norm[1],
                            bbox_norm[2], bbox_norm[3]
                        )
                        manip_cfg.setResize(
                            config.CLASSIFIER_INPUT_SIZE,
                            config.CLASSIFIER_INPUT_SIZE
                        )
                        q_manip_cfg.send(manip_cfg)

                        # Wait for classification result (with timeout)
                        in_class = q_class.tryGet()
                        if in_class is not None:
                            is_healthy, h_prob, u_prob = parse_classifier_output(in_class)

                            if is_healthy:
                                label = "HEALTHY"
                                color = config.COLOR_HEALTHY
                                class_conf = h_prob
                                count_healthy += 1
                            else:
                                label = "UNHEALTHY"
                                color = config.COLOR_UNHEALTHY
                                class_conf = u_prob
                                count_unhealthy += 1

                            total_oranges += 1
                            detection_counter += 1

                            # 4. Size measurement (healthy oranges)
                            if is_healthy and size_calc is not None and depth_frame is not None:
                                diameter_mm, grade, depth_z = size_calc.calculate_diameter(
                                    bbox_pixel, depth_frame
                                )
                                if diameter_mm > 0:
                                    size_info = {
                                        'diameter_mm': diameter_mm,
                                        'grade': grade,
                                        'depth_mm': depth_z
                                    }

                            # 5. Log result
                            if log_writer is not None:
                                try:
                                    log_writer.writerow([
                                        datetime.now().isoformat(),
                                        detection_counter,
                                        f"{det_conf:.3f}",
                                        label,
                                        f"{h_prob:.3f}",
                                        f"{u_prob:.3f}",
                                        f"{size_info['diameter_mm']:.1f}" if size_info else "",
                                        size_info['grade'] if size_info else "",
                                        f"{size_info['depth_mm']:.1f}" if size_info else ""
                                    ])
                                except Exception:
                                    pass

                    # 6. Draw detection on frame
                    display.draw_detection(
                        frame, bbox_pixel, label, class_conf, color, size_info
                    )

            else:
                display.draw_no_detections(frame)

            # ---------------------------------------------------------------
            # TRACKING / LATCH LOGIC (COMMENTED — for future conveyor use)
            # ---------------------------------------------------------------
            # TODO: Enable when conveyor belt tracking is needed
            #
            # # Dictionary to store "worst case" status per tracked orange
            # # orange_history = {}  # { track_id: "HEALTHY" | "UNHEALTHY" }
            #
            # # If this orange was EVER seen as unhealthy → stays unhealthy
            # # is_permanently_unhealthy = (
            # #     track_id in orange_history
            # #     and orange_history[track_id] == "UNHEALTHY"
            # # )
            #
            # # Only classify if not already condemned
            # # if not is_permanently_unhealthy:
            # #     ... run classifier ...
            # #     if unhealthy:
            # #         orange_history[track_id] = "UNHEALTHY"
            # #     else:
            # #         if track_id not in orange_history:
            # #             orange_history[track_id] = "PENDING"
            # ---------------------------------------------------------------

            # 7. Draw dashboard and status bar
            display.draw_dashboard(frame, fps, total_oranges, count_healthy, count_unhealthy)

            depth_info_text = ""
            if depth_frame is not None:
                # Show center depth for reference
                center_d = depth_frame[depth_frame.shape[0] // 2, depth_frame.shape[1] // 2]
                depth_info_text = f"Center Depth: {center_d}mm"

            display.draw_status_bar(frame, status_text, status_color, depth_info_text)

            # Depth overlay
            if show_depth_overlay and depth_frame is not None:
                _overlay_depth(frame, depth_frame)

            # 8. Show frame
            cv2.imshow(config.WINDOW_NAME, frame)

            # 9. Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Quit] Shutting down...")
                break
            elif key == ord('r'):
                total_oranges = 0
                count_healthy = 0
                count_unhealthy = 0
                detection_counter = 0
                print("[Reset] Counters cleared")
            elif key == ord('s'):
                screenshot_path = os.path.join(
                    config.LOG_DIR,
                    f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
                )
                os.makedirs(config.LOG_DIR, exist_ok=True)
                cv2.imwrite(screenshot_path, frame)
                print(f"[Screenshot] Saved: {screenshot_path}")
            elif key == ord('d'):
                show_depth_overlay = not show_depth_overlay
                print(f"[Depth] Overlay: {'ON' if show_depth_overlay else 'OFF'}")

        # ----------------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------------
        cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()

        print("\n" + "=" * 55)
        print("  SESSION SUMMARY")
        print(f"  Total Processed : {total_oranges}")
        print(f"  Healthy         : {count_healthy}")
        print(f"  Unhealthy       : {count_unhealthy}")
        print("=" * 55)


def _build_preview_pipeline():
    """Build a minimal pipeline for preview-only mode."""
    pipeline = dai.Pipeline()

    # RGB Camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(config.RGB_FPS)
    cam_rgb.setPreviewSize(config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT)
    cam_rgb.setInterleaved(False)

    # Mono cameras + stereo (for depth preview)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setExtendedDisparity(config.STEREO_EXTENDED_DISPARITY)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Preview output
    xout_preview = pipeline.create(dai.node.XLinkOut)
    xout_preview.setStreamName("preview")
    cam_rgb.preview.link(xout_preview.input)

    # Depth output
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def _overlay_depth(frame, depth_frame):
    """Overlay a colorized depth map on the frame (semi-transparent)."""
    # Normalize depth to 0-255 range (clip at 2 meters max)
    depth_vis = np.clip(depth_frame, 0, 2000)
    depth_vis = (depth_vis / 2000 * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Resize depth to match frame size
    depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))

    # Blend with original frame
    cv2.addWeighted(frame, 0.6, depth_color, 0.4, 0, frame)


if __name__ == "__main__":
    main()
