import math
from typing import Tuple, Optional
from enum import Enum


class Direction(Enum):
    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"


class NavigationSystem:
    def __init__(self, angle_threshold: float = 30.0):
        """
        Initialize navigation system

        Args:
            angle_threshold: Threshold angle in degrees for determining turns
        """
        self.angle_threshold = angle_threshold

    def determine_direction(self, user_direction: Tuple[float, float],
                          crosswalk_vector: Tuple[float, float],
                          crosswalk_position: Tuple[float, float],
                          image_center: Tuple[float, float]) -> Direction:
        """
        Determine navigation direction based on user heading and crosswalk orientation

        Args:
            user_direction: User's current direction vector (dx, dy)
            crosswalk_vector: Crosswalk orientation vector (dx, dy)
            crosswalk_position: Position of crosswalk center (x, y)
            image_center: Center of image/camera view (x, y)

        Returns:
            Direction enum indicating recommended action
        """
        # Calculate angle between user direction and crosswalk
        angle_to_crosswalk = self._angle_between_vectors(user_direction, crosswalk_vector)

        # Calculate position relative to user's path
        relative_pos = self._calculate_relative_position(
            crosswalk_position, image_center, user_direction
        )

        return self._classify_direction(angle_to_crosswalk, relative_pos)

    def determine_direction_simple(self, crosswalk_position: Tuple[float, float],
                                 image_center: Tuple[float, float],
                                 user_target_direction: str = "forward") -> Direction:
        """
        Simplified direction determination to keep user within crosswalk

        Args:
            crosswalk_position: Center of detected crosswalk (x, y)
            image_center: Center of camera view (x, y)
            user_target_direction: Desired direction ("forward", "left", "right")

        Returns:
            Direction enum indicating how to stay within crosswalk
        """
        cx, cy = crosswalk_position
        ix, iy = image_center

        # Calculate horizontal offset from crosswalk center
        horizontal_offset = cx - ix
        image_width = ix * 2  # Assuming center is at half width

        # Normalize offset (-1 to 1)
        normalized_offset = horizontal_offset / (image_width / 2)

        # Guide user to stay within crosswalk boundaries
        if abs(normalized_offset) < 0.15:  # Already well-centered in crosswalk
            return Direction.STRAIGHT
        elif normalized_offset > 0.15:  # Need to move left to stay in crosswalk
            return Direction.LEFT
        else:  # Need to move right to stay in crosswalk
            return Direction.RIGHT

    def calculate_navigation_instruction(self, direction: Direction,
                                       crosswalk_position: Tuple[float, float],
                                       image_center: Tuple[float, float]) -> dict:
        """
        Generate detailed navigation instruction

        Args:
            direction: Determined direction
            crosswalk_position: Center of crosswalk
            image_center: Center of image

        Returns:
            Dictionary with navigation details
        """
        cx, cy = crosswalk_position
        ix, iy = image_center

        # Calculate distance metrics (in pixels, could be converted to real units)
        distance = math.sqrt((cx - ix)**2 + (cy - iy)**2)
        horizontal_distance = abs(cx - ix)
        vertical_distance = abs(cy - iy)

        # Generate instruction text
        instruction_text = self._generate_instruction_text(
            direction, horizontal_distance, vertical_distance
        )

        return {
            "direction": direction.value,
            "instruction": instruction_text,
            "crosswalk_position": crosswalk_position,
            "distance_to_crosswalk": distance,
            "horizontal_offset": cx - ix,
            "vertical_offset": cy - iy,
            "confidence": self._calculate_confidence(distance, horizontal_distance)
        }

    def _angle_between_vectors(self, v1: Tuple[float, float],
                             v2: Tuple[float, float]) -> float:
        """Calculate angle between two vectors in degrees"""
        x1, y1 = v1
        x2, y2 = v2

        dot_product = x1 * x2 + y1 * y2
        mag1 = math.sqrt(x1*x1 + y1*y1)
        mag2 = math.sqrt(x2*x2 + y2*y2)

        if mag1 == 0 or mag2 == 0:
            return 0

        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        angle_rad = math.acos(cos_angle)

        # Determine sign using cross product
        cross_product = x1 * y2 - y1 * x2
        angle_deg = math.degrees(angle_rad)
        if cross_product < 0:
            angle_deg = -angle_deg

        return angle_deg

    def _calculate_relative_position(self, crosswalk_pos: Tuple[float, float],
                                   image_center: Tuple[float, float],
                                   user_direction: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate crosswalk position relative to user's path"""
        cx, cy = crosswalk_pos
        ix, iy = image_center
        ux, uy = user_direction

        # Vector from image center to crosswalk
        to_crosswalk = (cx - ix, cy - iy)

        # Project onto user direction and perpendicular
        magnitude = math.sqrt(ux*ux + uy*uy)
        if magnitude == 0:
            return (0, 0)

        # Normalize user direction
        ux_norm = ux / magnitude
        uy_norm = uy / magnitude

        # Perpendicular vector (rotated 90 degrees)
        perp_x = -uy_norm
        perp_y = ux_norm

        # Project crosswalk position onto these axes
        forward_proj = to_crosswalk[0] * ux_norm + to_crosswalk[1] * uy_norm
        lateral_proj = to_crosswalk[0] * perp_x + to_crosswalk[1] * perp_y

        return (lateral_proj, forward_proj)

    def _classify_direction(self, angle: float, relative_pos: Tuple[float, float]) -> Direction:
        """Classify direction based on angle and relative position"""
        lateral_offset, forward_distance = relative_pos

        # Primary classification based on angle
        if abs(angle) <= self.angle_threshold:
            # Crosswalk is roughly aligned with user direction
            if abs(lateral_offset) < 50:  # Small threshold for "straight"
                return Direction.STRAIGHT
            elif lateral_offset > 0:
                return Direction.RIGHT
            else:
                return Direction.LEFT

        elif angle > self.angle_threshold:
            # Need to turn right to align with crosswalk
            return Direction.RIGHT

        else:
            # Need to turn left to align with crosswalk
            return Direction.LEFT

    def _generate_instruction_text(self, direction: Direction,
                                 horizontal_distance: float,
                                 vertical_distance: float) -> str:
        """Generate human-readable navigation instruction for staying in crosswalk"""
        if direction == Direction.STRAIGHT:
            if vertical_distance > 100:
                return "Continue straight towards the crosswalk"
            else:
                return "Stay centered in the crosswalk"
        elif direction == Direction.LEFT:
            if vertical_distance > 100:
                return "Move slightly left to align with the crosswalk"
            else:
                return "Step left to stay within the crosswalk"
        elif direction == Direction.RIGHT:
            if vertical_distance > 100:
                return "Move slightly right to align with the crosswalk"
            else:
                return "Step right to stay within the crosswalk"

        return "Follow the crosswalk"

    def _calculate_confidence(self, distance: float, horizontal_distance: float) -> float:
        """Calculate confidence score for the navigation instruction"""
        # Confidence decreases with distance and horizontal offset
        distance_factor = max(0, 1 - distance / 500)  # Normalize distance
        offset_factor = max(0, 1 - horizontal_distance / 200)  # Normalize offset

        confidence = (distance_factor + offset_factor) / 2
        return max(0.1, min(1.0, confidence))