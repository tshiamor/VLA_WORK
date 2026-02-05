"""
Termination functions for VLA tasks.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Terminate episode when max steps reached.

    Args:
        env: Environment instance.

    Returns:
        Termination mask [num_envs].
    """
    return env.episode_length_buf >= env.max_episode_length


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Terminate if object falls below minimum height.

    Args:
        env: Environment instance.
        minimum_height: Minimum allowed height.
        asset_cfg: Asset configuration.

    Returns:
        Termination mask [num_envs].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def object_lifted_success(
    env: ManagerBasedRLEnv,
    success_height: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Terminate with success when object lifted above threshold.

    Args:
        env: Environment instance.
        success_height: Height threshold for success.
        asset_cfg: Asset configuration.

    Returns:
        Termination mask [num_envs].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > success_height
