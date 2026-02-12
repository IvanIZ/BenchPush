from enum import Enum

class ObstacleType(Enum):
    """Obstacle type enumeration."""
    STATIC = 0
    DYNAMIC = 1
    BOUNDARY = 2