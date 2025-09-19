import numpy as np
import math
from typing import Tuple, List, Optional
import cv2


class VectorCalculator:
    def __init__(self):
        """Initialize vector calculator for crosswalk analysis"""
        pass

    def calculate_crosswalk_vector(self, crosswalk_lines: List[np.ndarray],
                                 image_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        Calculate the main direction vector of the crosswalk

        Args:
            crosswalk_lines: List of line segments representing crosswalk stripes
            image_shape: (height, width) of the image

        Returns:
            Normalized direction vector (dx, dy) or None if calculation fails
        """
        if not crosswalk_lines:
            return None

        # Calculate average direction of all crosswalk lines
        total_dx = 0
        total_dy = 0
        valid_lines = 0

        for line in crosswalk_lines:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1

            # Skip very short lines
            length = math.sqrt(dx*dx + dy*dy)
            if length < 10:
                continue

            # Normalize and accumulate
            total_dx += dx / length
            total_dy += dy / length
            valid_lines += 1

        if valid_lines == 0:
            return None

        # Average direction
        avg_dx = total_dx / valid_lines
        avg_dy = total_dy / valid_lines

        # Normalize the result
        magnitude = math.sqrt(avg_dx*avg_dx + avg_dy*avg_dy)
        if magnitude > 0:
            return (avg_dx / magnitude, avg_dy / magnitude)

        return None

    def calculate_crosswalk_vector_from_bbox(self, bbox: np.ndarray,
                                           image: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Calculate crosswalk vector from bounding box using edge detection

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image: Input image

        Returns:
            Normalized direction vector (dx, dy) or None if calculation fails
        """
        x1, y1, x2, y2 = bbox.astype(int)

        # Extract ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Use Hough transform to find dominant lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=15,
            maxLineGap=5
        )

        if lines is not None and len(lines) > 0:
            return self.calculate_crosswalk_vector(lines, roi.shape[:2])

        return None

    def get_crosswalk_center(self, crosswalk_data: dict) -> Tuple[float, float]:
        """
        Get the center point of the crosswalk

        Args:
            crosswalk_data: Dictionary containing crosswalk detection data

        Returns:
            Center coordinates (x, y)
        """
        if 'center' in crosswalk_data:
            return crosswalk_data['center']
        elif 'bbox' in crosswalk_data:
            bbox = crosswalk_data['bbox']
            x1, y1, x2, y2 = bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        else:
            raise ValueError("Invalid crosswalk data format")

    def calculate_relative_position(self, crosswalk_center: Tuple[float, float],
                                  image_center: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate relative position of crosswalk to image center

        Args:
            crosswalk_center: Center of detected crosswalk
            image_center: Center of the image (camera view)

        Returns:
            Relative position vector (dx, dy)
        """
        cx, cy = crosswalk_center
        ix, iy = image_center

        return (cx - ix, cy - iy)

    def angle_between_vectors(self, v1: Tuple[float, float],
                            v2: Tuple[float, float]) -> float:
        """
        Calculate angle between two vectors in degrees

        Args:
            v1: First vector (dx, dy)
            v2: Second vector (dx, dy)

        Returns:
            Angle in degrees (-180 to 180)
        """
        x1, y1 = v1
        x2, y2 = v2

        # Calculate dot product and magnitudes
        dot_product = x1 * x2 + y1 * y2
        mag1 = math.sqrt(x1*x1 + y1*y1)
        mag2 = math.sqrt(x2*x2 + y2*y2)

        if mag1 == 0 or mag2 == 0:
            return 0

        # Calculate angle
        cos_angle = dot_product / (mag1 * mag2)
        # Clamp to valid range for arccos
        cos_angle = max(-1, min(1, cos_angle))
        angle_rad = math.acos(cos_angle)

        # Convert to degrees
        angle_deg = math.degrees(angle_rad)

        # Determine sign using cross product
        cross_product = x1 * y2 - y1 * x2
        if cross_product < 0:
            angle_deg = -angle_deg

        return angle_deg

    def get_perpendicular_vector(self, vector: Tuple[float, float]) -> Tuple[float, float]:
        """
        Get perpendicular vector (90 degrees rotation)

        Args:
            vector: Input vector (dx, dy)

        Returns:
            Perpendicular vector
        """
        dx, dy = vector
        return (-dy, dx)