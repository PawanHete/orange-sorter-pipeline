"""
==========================================================
 SIZE CALCULATOR — Stereo Depth → Real-World Diameter (mm)
==========================================================
 Uses the OAK-D Pro W stereo depth map + camera intrinsics
 to calculate the real-world size of detected oranges.

 Method: Pinhole Camera Model
   Real_Size = (Pixel_Size × Depth_Z) / Focal_Length
==========================================================
"""

import numpy as np
from . import config


class SizeCalculator:
    """
    Calculates real-world object dimensions from stereo depth data
    and camera calibration intrinsics.
    """

    def __init__(self, device):
        """
        Read camera intrinsics from the OAK-D Pro W calibration data.

        Args:
            device: dai.Device — the connected OAK device
        """
        import depthai as dai

        calib = device.readCalibration()

        # Get RGB camera intrinsics (since depth is aligned to RGB)
        # Intrinsic matrix is 3x3:
        # [[fx,  0, cx],
        #  [ 0, fy, cy],
        #  [ 0,  0,  1]]
        intrinsics = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            config.PREVIEW_WIDTH,
            config.PREVIEW_HEIGHT
        )

        self.fx = intrinsics[0][0]  # Focal length X (pixels)
        self.fy = intrinsics[1][1]  # Focal length Y (pixels)
        self.cx = intrinsics[0][2]  # Principal point X
        self.cy = intrinsics[1][2]  # Principal point Y

        print(f"[SizeCalculator] Camera intrinsics loaded:")
        print(f"  fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def calculate_diameter(self, bbox_pixel, depth_frame):
        """
        Calculate the real-world diameter of an orange from its bounding box
        and the stereo depth map.

        Args:
            bbox_pixel: tuple (x1, y1, x2, y2) — pixel coordinates in the
                        preview frame (PREVIEW_WIDTH × PREVIEW_HEIGHT)
            depth_frame: np.ndarray — depth map from StereoDepth node,
                         aligned to RGB. Values are in millimeters (uint16).

        Returns:
            tuple: (diameter_mm, grade_label, depth_z_mm)
                - diameter_mm: float — estimated diameter in mm, or -1 if invalid
                - grade_label: str — "SMALL", "MEDIUM", "LARGE", or "N/A"
                - depth_z_mm: float — depth at center in mm, or -1 if invalid
        """
        x1, y1, x2, y2 = bbox_pixel

        # Bounding box dimensions in pixels
        bbox_width_px = x2 - x1
        bbox_height_px = y2 - y1

        if bbox_width_px <= 0 or bbox_height_px <= 0:
            return -1.0, "N/A", -1.0

        # -------------------------------------------------------
        # Extract depth Z from the center region of the bbox
        # -------------------------------------------------------
        # We sample the center portion to avoid edge noise
        shrink = config.DEPTH_ROI_SHRINK_FACTOR
        center_x1 = int(x1 + bbox_width_px * shrink)
        center_y1 = int(y1 + bbox_height_px * shrink)
        center_x2 = int(x2 - bbox_width_px * shrink)
        center_y2 = int(y2 - bbox_height_px * shrink)

        # Clamp to depth frame dimensions
        dh, dw = depth_frame.shape[:2]
        # Scale bbox coordinates if depth frame has different resolution
        scale_x = dw / config.PREVIEW_WIDTH
        scale_y = dh / config.PREVIEW_HEIGHT

        d_x1 = max(0, int(center_x1 * scale_x))
        d_y1 = max(0, int(center_y1 * scale_y))
        d_x2 = min(dw - 1, int(center_x2 * scale_x))
        d_y2 = min(dh - 1, int(center_y2 * scale_y))

        if d_x2 <= d_x1 or d_y2 <= d_y1:
            return -1.0, "N/A", -1.0

        # Extract depth ROI
        depth_roi = depth_frame[d_y1:d_y2, d_x1:d_x2]

        # Filter out zero/invalid depth values
        valid_depths = depth_roi[depth_roi > 0]

        if len(valid_depths) == 0:
            return -1.0, "N/A", -1.0

        # Use median depth for robustness (resists outliers)
        depth_z_mm = float(np.median(valid_depths))

        # -------------------------------------------------------
        # Calculate real-world size using pinhole camera model
        # -------------------------------------------------------
        # Real_Size_mm = (Pixel_Size × Depth_mm) / Focal_Length_px
        width_mm = (bbox_width_px * depth_z_mm) / self.fx
        height_mm = (bbox_height_px * depth_z_mm) / self.fy

        # Orange is approximately spherical → average width and height
        diameter_mm = (width_mm + height_mm) / 2.0

        # -------------------------------------------------------
        # Grade the orange by size
        # -------------------------------------------------------
        grade = self._grade_orange(diameter_mm)

        return diameter_mm, grade, depth_z_mm

    @staticmethod
    def _grade_orange(diameter_mm):
        """
        Grade an orange based on its diameter.

        Returns:
            str: "SMALL", "MEDIUM", or "LARGE"
        """
        if diameter_mm < config.SIZE_SMALL_MAX:
            return "SMALL"
        elif diameter_mm <= config.SIZE_MEDIUM_MAX:
            return "MEDIUM"
        else:
            return "LARGE"
