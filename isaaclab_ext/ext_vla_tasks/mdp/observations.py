"""
Observation functions for VLA tasks.

Includes camera observations and proprioceptive state functions.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def camera_rgb(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Get RGB image from camera sensor.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the camera sensor.

    Returns:
        RGB image tensor [num_envs, H, W, 3] normalized to [0, 1].
    """
    camera: Camera = env.scene[sensor_cfg.name]
    # Get RGB data and normalize to [0, 1]
    rgb = camera.data.output["rgb"].clone()

    # Handle RGBA -> RGB if needed
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    # Normalize to [0, 1] if not already
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0

    return rgb


def camera_depth(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    normalize: bool = True,
    max_depth: float = 5.0,
) -> torch.Tensor:
    """
    Get depth image from camera sensor.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the camera sensor.
        normalize: Whether to normalize depth to [0, 1].
        max_depth: Maximum depth for normalization.

    Returns:
        Depth image tensor [num_envs, H, W, 1].
    """
    camera: Camera = env.scene[sensor_cfg.name]
    depth = camera.data.output["depth"].clone()

    if normalize:
        depth = torch.clamp(depth / max_depth, 0.0, 1.0)

    return depth


def joint_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get joint positions relative to default.

    Args:
        env: The environment instance.

    Returns:
        Relative joint positions [num_envs, num_joints].
    """
    robot: Articulation = env.scene["robot"]
    return robot.data.joint_pos - robot.data.default_joint_pos


def joint_vel_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get joint velocities.

    Args:
        env: The environment instance.

    Returns:
        Joint velocities [num_envs, num_joints].
    """
    robot: Articulation = env.scene["robot"]
    return robot.data.joint_vel


def object_position_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get object position in robot root frame.

    Args:
        env: The environment instance.

    Returns:
        Object position [num_envs, 3].
    """
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]

    # Get robot root position and orientation
    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w

    # Get object position
    obj_pos = obj.data.root_pos_w

    # Transform to robot frame
    # Simplified: just subtract robot position (assumes robot is at origin with identity rotation)
    obj_pos_rel = obj_pos - robot_pos

    return obj_pos_rel


def ee_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get end-effector position in world frame.

    Args:
        env: The environment instance.

    Returns:
        EE position [num_envs, 3].
    """
    ee_frame = env.scene["ee_frame"]
    return ee_frame.data.target_pos_w[..., 0, :]


def ee_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get end-effector orientation (quaternion) in world frame.

    Args:
        env: The environment instance.

    Returns:
        EE orientation [num_envs, 4].
    """
    ee_frame = env.scene["ee_frame"]
    return ee_frame.data.target_quat_w[..., 0, :]


def gripper_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get gripper open/close state.

    Args:
        env: The environment instance.

    Returns:
        Gripper state [num_envs, 1] (0=closed, 1=open).
    """
    robot: Articulation = env.scene["robot"]
    # Get finger joint positions (assuming last 2 joints are fingers)
    finger_pos = robot.data.joint_pos[:, -2:]
    # Average finger opening
    gripper_opening = finger_pos.mean(dim=-1, keepdim=True)
    # Normalize (0.04 is fully open for Franka)
    return gripper_opening / 0.04


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get last action taken.

    Args:
        env: The environment instance.

    Returns:
        Last action [num_envs, action_dim].
    """
    return env.action_manager.action
