"""
==========================================================
 DISPLAY — OpenCV HUD Renderer
==========================================================
 Draws bounding boxes, labels, stats dashboard, and
 status bar on the preview frame.

 Layout:
 ┌──────────────────────────────────────────────┐
 │ FPS: 28.5 │ Total: 12 │ Healthy: 9 │ Bad: 3 │  ← Top bar
 ├──────────────────────────────────────────────┤
 │                                              │
 │      ┌──────────┐                            │
 │      │ GREEN BOX│ HEALTHY 92%                │
 │      │          │ D: 72mm (MEDIUM)           │
 │      └──────────┘                            │
 │                                              │
 ├──────────────────────────────────────────────┤
 │ STATUS: ORANGE DETECTED │ DEPTH: 342mm       │  ← Bottom bar
 └──────────────────────────────────────────────┘
==========================================================
"""

import cv2
from . import config


def draw_dashboard(frame, fps, total, healthy, unhealthy):
    """
    Draw the top stats bar.

    Args:
        frame: np.ndarray — the display frame (modified in-place)
        fps: float — current frames per second
        total: int — total oranges detected in session
        healthy: int — healthy count
        unhealthy: int — unhealthy count
    """
    h, w = frame.shape[:2]
    bar_height = 40

    # Black background bar
    cv2.rectangle(frame, (0, 0), (w, bar_height), config.COLOR_TEXT_BG, -1)

    # Stats text
    stats = f"FPS: {fps:.1f}  |  Total: {total}  |  Healthy: {healthy}  |  Unhealthy: {unhealthy}"
    cv2.putText(
        frame, stats, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT_WHITE, 1, cv2.LINE_AA
    )


def draw_detection(frame, bbox_pixel, label, confidence, color, size_info=None):
    """
    Draw a bounding box with label for a detected orange.

    Args:
        frame: np.ndarray — the display frame (modified in-place)
        bbox_pixel: tuple (x1, y1, x2, y2) — pixel coordinates
        label: str — "HEALTHY" or "UNHEALTHY"
        confidence: float — classification confidence (0-1)
        color: tuple — BGR color for the box
        size_info: dict or None — if provided, contains:
            - 'diameter_mm': float
            - 'grade': str
            - 'depth_mm': float
    """
    x1, y1, x2, y2 = bbox_pixel

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label with confidence
    label_text = f"{label} {confidence:.0%}"
    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Label background
    cv2.rectangle(
        frame,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0] + 4, y1),
        color, -1
    )
    cv2.putText(
        frame, label_text, (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_TEXT_WHITE, 1, cv2.LINE_AA
    )

    # Size info (only for healthy oranges)
    if size_info and size_info.get('diameter_mm', -1) > 0:
        diameter = size_info['diameter_mm']
        grade = size_info['grade']
        depth = size_info['depth_mm']

        size_text = f"D: {diameter:.0f}mm ({grade})"
        cv2.putText(
            frame, size_text, (x1, y2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_SIZE_TEXT, 1, cv2.LINE_AA
        )

        depth_text = f"Depth: {depth:.0f}mm"
        cv2.putText(
            frame, depth_text, (x1, y2 + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA
        )


def draw_status_bar(frame, status_text, status_color, depth_info=""):
    """
    Draw the bottom status bar.

    Args:
        frame: np.ndarray — the display frame (modified in-place)
        status_text: str — e.g., "ORANGE DETECTED" or "NO ORANGES"
        status_color: tuple — BGR color for the status text
        depth_info: str — optional depth info text
    """
    h, w = frame.shape[:2]
    bar_height = 35
    bar_top = h - bar_height

    # Black background bar
    cv2.rectangle(frame, (0, bar_top), (w, h), config.COLOR_TEXT_BG, -1)

    # Status text
    cv2.putText(
        frame, f"STATUS: {status_text}", (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA
    )

    # Depth info on the right side
    if depth_info:
        info_size, _ = cv2.getTextSize(depth_info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            frame, depth_info, (w - info_size[0] - 10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
        )


def draw_no_detections(frame):
    """Draw a subtle indicator when no oranges are visible."""
    h, w = frame.shape[:2]
    cv2.putText(
        frame, "Waiting for oranges...", (w // 2 - 100, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_NO_DETECTION, 1, cv2.LINE_AA
    )
