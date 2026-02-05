"""
Camera utilities for VLA
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])


class CameraUtils:
    """Utilities for camera operations in VLA."""

    @staticmethod
    def pixel_to_camera(
        uv: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        """
        Convert pixel coordinates to camera frame 3D points.

        Args:
            uv: Pixel coordinates [N, 2] (u, v)
            depth: Depth values [N]
            intrinsics: Camera intrinsics

        Returns:
            3D points in camera frame [N, 3]
        """
        u, v = uv[:, 0], uv[:, 1]

        x = (u - intrinsics.cx) * depth / intrinsics.fx
        y = (v - intrinsics.cy) * depth / intrinsics.fy
        z = depth

        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def camera_to_world(
        points_cam: np.ndarray,
        cam_pose: np.ndarray,
    ) -> np.ndarray:
        """
        Transform points from camera frame to world frame.

        Args:
            points_cam: Points in camera frame [N, 3]
            cam_pose: Camera pose (4x4 homogeneous transform)

        Returns:
            Points in world frame [N, 3]
        """
        # Add homogeneous coordinate
        points_homo = np.concatenate([
            points_cam,
            np.ones((points_cam.shape[0], 1))
        ], axis=-1)

        # Transform
        points_world = (cam_pose @ points_homo.T).T

        return points_world[:, :3]

    @staticmethod
    def depth_to_pointcloud(
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        rgb: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert depth image to point cloud.

        Args:
            depth: Depth image [H, W]
            intrinsics: Camera intrinsics
            rgb: Optional RGB image [H, W, 3]
            mask: Optional valid pixel mask [H, W]

        Returns:
            points: Point cloud [N, 3]
            colors: Optional colors [N, 3]
        """
        H, W = depth.shape

        # Create pixel grid
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()
        z = depth.flatten()

        # Apply mask if provided
        if mask is not None:
            valid = mask.flatten() & (z > 0)
        else:
            valid = z > 0

        u = u[valid]
        v = v[valid]
        z = z[valid]

        # Convert to 3D
        x = (u - intrinsics.cx) * z / intrinsics.fx
        y = (v - intrinsics.cy) * z / intrinsics.fy

        points = np.stack([x, y, z], axis=-1)

        # Get colors
        colors = None
        if rgb is not None:
            colors = rgb.reshape(-1, 3)[valid]

        return points, colors

    @staticmethod
    def project_points(
        points: np.ndarray,
        intrinsics: CameraIntrinsics,
        cam_pose_inv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Project 3D points to image plane.

        Args:
            points: 3D points [N, 3] in camera frame (or world if pose given)
            intrinsics: Camera intrinsics
            cam_pose_inv: Inverse camera pose (world to camera)

        Returns:
            Pixel coordinates [N, 2]
        """
        # Transform to camera frame if pose given
        if cam_pose_inv is not None:
            points_homo = np.concatenate([
                points,
                np.ones((points.shape[0], 1))
            ], axis=-1)
            points = (cam_pose_inv @ points_homo.T).T[:, :3]

        # Project
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        u = intrinsics.fx * x / z + intrinsics.cx
        v = intrinsics.fy * y / z + intrinsics.cy

        return np.stack([u, v], axis=-1)

    @staticmethod
    def resize_intrinsics(
        intrinsics: CameraIntrinsics,
        new_width: int,
        new_height: int,
    ) -> CameraIntrinsics:
        """
        Adjust intrinsics for resized image.

        Args:
            intrinsics: Original intrinsics
            new_width: New image width
            new_height: New image height

        Returns:
            Adjusted intrinsics
        """
        scale_x = new_width / intrinsics.width
        scale_y = new_height / intrinsics.height

        return CameraIntrinsics(
            fx=intrinsics.fx * scale_x,
            fy=intrinsics.fy * scale_y,
            cx=intrinsics.cx * scale_x,
            cy=intrinsics.cy * scale_y,
            width=new_width,
            height=new_height,
        )

    @staticmethod
    def get_default_intrinsics(
        width: int = 224,
        height: int = 224,
        fov_deg: float = 60.0,
    ) -> CameraIntrinsics:
        """
        Create default intrinsics for a given resolution and FOV.

        Args:
            width: Image width
            height: Image height
            fov_deg: Horizontal field of view in degrees

        Returns:
            Camera intrinsics
        """
        fov_rad = np.deg2rad(fov_deg)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Square pixels

        return CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=width / 2,
            cy=height / 2,
            width=width,
            height=height,
        )


class DepthProcessor:
    """Process depth images for VLA."""

    def __init__(
        self,
        near: float = 0.01,
        far: float = 10.0,
        normalize: bool = True,
    ):
        self.near = near
        self.far = far
        self.normalize = normalize

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        """Process depth image."""
        # Clip to valid range
        depth = np.clip(depth, self.near, self.far)

        # Normalize to [0, 1]
        if self.normalize:
            depth = (depth - self.near) / (self.far - self.near)

        return depth

    def denormalize(self, depth: np.ndarray) -> np.ndarray:
        """Convert normalized depth back to meters."""
        if self.normalize:
            depth = depth * (self.far - self.near) + self.near
        return depth
