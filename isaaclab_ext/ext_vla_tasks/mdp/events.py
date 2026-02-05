"""
Event functions for VLA tasks.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """
    Reset scene entities to their default states.

    Args:
        env: Environment instance.
        env_ids: Environment indices to reset.
    """
    # Reset robot
    robot: Articulation = env.scene["robot"]
    default_joint_pos = robot.data.default_joint_pos[env_ids]
    default_joint_vel = torch.zeros_like(default_joint_pos)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    # Reset object
    obj: RigidObject = env.scene["object"]
    default_root_state = obj.data.default_root_state[env_ids].clone()
    obj.write_root_state_to_sim(default_root_state, env_ids=env_ids)


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict,
    velocity_range: dict,
    asset_cfg: SceneEntityCfg,
):
    """
    Reset object root state with uniform random sampling.

    Args:
        env: Environment instance.
        env_ids: Environment indices to reset.
        pose_range: Dict with x, y, z ranges for position randomization.
        velocity_range: Dict with velocity ranges (usually empty).
        asset_cfg: Asset configuration.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get default state
    root_state = asset.data.default_root_state[env_ids].clone()

    # Randomize position
    if "x" in pose_range:
        root_state[:, 0] += sample_uniform(
            pose_range["x"][0], pose_range["x"][1], (len(env_ids),), device=env.device
        )
    if "y" in pose_range:
        root_state[:, 1] += sample_uniform(
            pose_range["y"][0], pose_range["y"][1], (len(env_ids),), device=env.device
        )
    if "z" in pose_range:
        root_state[:, 2] += sample_uniform(
            pose_range["z"][0], pose_range["z"][1], (len(env_ids),), device=env.device
        )

    # Write to simulation
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
