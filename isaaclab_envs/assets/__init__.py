"""Robot assets with camera configurations"""

from .franka_with_cameras import FrankaWithCameras, FRANKA_WITH_CAMERAS_CFG
from .ur5e_with_cameras import UR5eWithCameras, UR5E_WITH_CAMERAS_CFG

__all__ = [
    "FrankaWithCameras",
    "FRANKA_WITH_CAMERAS_CFG",
    "UR5eWithCameras",
    "UR5E_WITH_CAMERAS_CFG",
]
