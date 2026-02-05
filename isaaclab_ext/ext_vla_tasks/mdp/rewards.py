"""
Reward functions for VLA tasks.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Reward for reaching the object.

    Gaussian reward based on distance between EE and object.

    Args:
        env: Environment instance.
        std: Standard deviation for Gaussian reward.
        object_cfg: Object configuration.

    Returns:
        Reward tensor [num_envs].
    """
    # Get end-effector position
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    # Get object position
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w

    # Compute distance
    distance = torch.norm(ee_pos - obj_pos, dim=-1)

    # Gaussian reward
    return torch.exp(-distance / std)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Reward for lifting the object above a threshold.

    Args:
        env: Environment instance.
        minimal_height: Minimum height for reward.
        object_cfg: Object configuration.

    Returns:
        Reward tensor [num_envs].
    """
    obj: RigidObject = env.scene[object_cfg.name]
    # Height relative to table (table is at z=0)
    height = obj.data.root_pos_w[:, 2]
    return (height > minimal_height).float()


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    goal_pos: tuple = (0.5, 0.0, 0.3),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Reward for moving object to goal position.

    Args:
        env: Environment instance.
        std: Standard deviation for Gaussian reward.
        goal_pos: Target position for object.
        object_cfg: Object configuration.

    Returns:
        Reward tensor [num_envs].
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w

    goal = torch.tensor(goal_pos, device=obj_pos.device).unsqueeze(0)
    distance = torch.norm(obj_pos - goal, dim=-1)

    return torch.exp(-distance / std)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalty for large changes in actions (smoothness).

    Args:
        env: Environment instance.

    Returns:
        Penalty tensor [num_envs].
    """
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)


def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalty for high joint velocities.

    Args:
        env: Environment instance.
        asset_cfg: Robot configuration.

    Returns:
        Penalty tensor [num_envs].
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_vel), dim=-1)
