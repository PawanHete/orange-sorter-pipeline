"""
==========================================================
 YOLO11 POST-PROCESSING
==========================================================
 Parses the raw YOLO11n output tensor [1, 5, 8400] into
 bounding boxes with confidence scores.

 YOLO11 is anchor-free:
   - 8400 detection candidates
   - 5 values per candidate: [cx, cy, w, h, conf]
   - Coordinates are in pixel space of the 640x640 input
==========================================================
"""

import numpy as np
from . import config


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def nms(boxes, scores, iou_threshold):
    """
    Non-Maximum Suppression.

    Args:
        boxes: np.array of shape [N, 4] — (x1, y1, x2, y2)
        scores: np.array of shape [N]
        iou_threshold: float

    Returns:
        List of indices to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def parse_yolo_output(raw_output, frame_width, frame_height):
    """
    Parse YOLO11n raw output tensor into detections.

    Args:
        raw_output: np.array of shape [1, 5, 8400]
            - 5 = [cx, cy, w, h, class_score]
            - For single-class model, class_score is the confidence
        frame_width: int — width of the input frame (for normalization)
        frame_height: int — height of the input frame (for normalization)

    Returns:
        List of dicts, each with keys:
            - 'bbox': (x1, y1, x2, y2) normalized to [0, 1]
            - 'confidence': float
            - 'bbox_pixel': (x1, y1, x2, y2) in pixel coords of frame_width/height
    """
    # raw_output shape: [1, 5, 8400] → squeeze → [5, 8400]
    output = raw_output.squeeze(0)  # [5, 8400]

    # Transpose to [8400, 5] for easier indexing
    output = output.T  # [8400, 5]

    # Extract components
    cx = output[:, 0]  # Center X (in 640x640 space)
    cy = output[:, 1]  # Center Y
    w = output[:, 2]   # Width
    h = output[:, 3]   # Height
    scores = output[:, 4]  # Class score (single class)

    # Apply sigmoid to scores (YOLO11 raw outputs are logits)
    confidences = sigmoid(scores)

    # Filter by confidence threshold
    mask = confidences > config.YOLO_CONFIDENCE_THRESHOLD
    cx, cy, w, h, confidences = cx[mask], cy[mask], w[mask], h[mask], confidences[mask]

    if len(confidences) == 0:
        return []

    # Convert center format to corner format (x1, y1, x2, y2)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Stack for NMS
    boxes = np.stack([x1, y1, x2, y2], axis=1)  # [N, 4] in 640x640 space

    # Apply NMS
    keep = nms(boxes, confidences, config.YOLO_IOU_THRESHOLD)
    boxes = boxes[keep]
    confidences = confidences[keep]

    # Build results
    detections = []
    for i in range(len(keep)):
        # Normalize to [0, 1] range (relative to YOLO input 640x640)
        x1_norm = boxes[i, 0] / config.YOLO_INPUT_SIZE
        y1_norm = boxes[i, 1] / config.YOLO_INPUT_SIZE
        x2_norm = boxes[i, 2] / config.YOLO_INPUT_SIZE
        y2_norm = boxes[i, 3] / config.YOLO_INPUT_SIZE

        # Clamp to [0, 1]
        x1_norm = max(0.0, min(1.0, x1_norm))
        y1_norm = max(0.0, min(1.0, y1_norm))
        x2_norm = max(0.0, min(1.0, x2_norm))
        y2_norm = max(0.0, min(1.0, y2_norm))

        # Pixel coordinates on the actual frame
        px1 = int(x1_norm * frame_width)
        py1 = int(y1_norm * frame_height)
        px2 = int(x2_norm * frame_width)
        py2 = int(y2_norm * frame_height)

        detections.append({
            'bbox': (x1_norm, y1_norm, x2_norm, y2_norm),
            'confidence': float(confidences[i]),
            'bbox_pixel': (px1, py1, px2, py2),
        })

    return detections
