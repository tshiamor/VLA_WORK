"""
UR5e VLA Environment

Isaac Lab environment for training VLA models on Universal Robots UR5e.
Provides camera observations, proprioceptive state, and action interfaces.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import Camera, CameraCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.utils import configclass

from ..assets.ur5e_with_cameras import (
    UR5E_ARTICULATION_CFG,
    UR5E_CONTEXT_CAMERA_CFG,
    UR5E_WRIST_CAMERA_CFG,
    UR5E_WITH_CAMERAS_CFG,
)


@configclass
class UR5eSceneCfg(InteractiveSceneCfg):
    """Configuration for UR5e scene with cameras."""

    # Ground plane
    ground = sim_utils.GroundPlaneCfg()

    # Lighting
    dome_light = sim_utils.DomeLightCfg(
        intensity=1500.0,
        color=(0.9, 0.9, 0.9),
    )

    # Robot
    robot: ArticulationCfg = UR5E_ARTICULATION_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Cameras
    context_camera: CameraCfg = UR5E_CONTEXT_CAMERA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ContextCamera"
    )
    wrist_camera: CameraCfg = UR5E_WRIST_CAMERA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/WristCamera"
    )


def ur5e_joint_position_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get UR5e joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def ur5e_joint_velocity_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get UR5e joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def ur5e_ee_position_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get end-effector position in world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_idx = asset.find_bodies("ee_link")[0][0]
    return asset.data.body_pos_w[:, ee_body_idx, :]


def ur5e_ee_orientation_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get end-effector orientation (quaternion) in world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_idx = asset.find_bodies("ee_link")[0][0]
    return asset.data.body_quat_w[:, ee_body_idx, :]


def ur5e_context_camera_obs(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get context camera RGB image."""
    camera: Camera = env.scene[sensor_cfg.name]
    return camera.data.output["rgb"][..., :3]


def ur5e_wrist_camera_obs(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get wrist camera RGB image."""
    camera: Camera = env.scene[sensor_cfg.name]
    return camera.data.output["rgb"][..., :3]


@configclass
class UR5eObservationsCfg:
    """Observation configuration for UR5e VLA."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy (proprioceptive state)."""

        joint_pos = ObsTerm(
            func=ur5e_joint_position_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=ur5e_joint_velocity_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        ee_position = ObsTerm(
            func=ur5e_ee_position_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        ee_orientation = ObsTerm(
            func=ur5e_ee_orientation_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CameraCfg(ObsGroup):
        """Camera observations for VLA."""

        context_rgb = ObsTerm(
            func=ur5e_context_camera_obs,
            params={"sensor_cfg": SceneEntityCfg("context_camera")},
        )
        wrist_rgb = ObsTerm(
            func=ur5e_wrist_camera_obs,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    camera: CameraCfg = CameraCfg()


@configclass
class UR5eVLAEnvCfg(ManagerBasedEnvCfg):
    """Configuration for UR5e VLA environment."""

    # Scene
    scene: UR5eSceneCfg = UR5eSceneCfg(num_envs=1, env_spacing=2.5)

    # Observations
    observations: UR5eObservationsCfg = UR5eObservationsCfg()

    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120.0,
        render_interval=1,
    )

    # Episode settings
    episode_length_s = 10.0


class UR5eVLAEnv(ManagerBasedEnv):
    """
    UR5e environment for VLA training.

    Provides:
    - RGB observations from context and wrist cameras
    - Proprioceptive state (joint positions, velocities, EE pose)
    - Joint position action interface
    """

    cfg: UR5eVLAEnvCfg

    def __init__(self, cfg: UR5eVLAEnvCfg, render_mode: str = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache robot reference
        self._robot: Articulation = self.scene["robot"]

        # Action parameters
        self._action_dim = 6  # 6 arm joints
        self._action_scale = 0.1

    @property
    def action_dim(self) -> int:
        """Return action dimension."""
        return self._action_dim

    @property
    def proprio_dim(self) -> int:
        """Return proprioceptive state dimension."""
        return 12  # 6 pos + 6 vel

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Get current observations.

        Returns:
            Dictionary containing:
            - 'proprio': Proprioceptive state [num_envs, 12]
            - 'context_rgb': Context camera image [num_envs, H, W, 3]
            - 'wrist_rgb': Wrist camera image [num_envs, H, W, 3]
        """
        obs = {}

        # Proprioceptive state
        joint_pos = self._robot.data.joint_pos[:, :6]
        joint_vel = self._robot.data.joint_vel[:, :6]
        obs['proprio'] = torch.cat([joint_pos, joint_vel], dim=-1)

        # Camera observations
        context_cam: Camera = self.scene["context_camera"]
        wrist_cam: Camera = self.scene["wrist_camera"]

        obs['context_rgb'] = context_cam.data.output["rgb"][..., :3]
        obs['wrist_rgb'] = wrist_cam.data.output["rgb"][..., :3]

        return obs

    def apply_action(self, actions: torch.Tensor) -> None:
        """
        Apply joint position actions to the robot.

        Args:
            actions: Joint position targets [num_envs, 6]
        """
        # Clamp actions to joint limits
        joint_limits = UR5E_WITH_CAMERAS_CFG.joint_limits
        lower = torch.tensor([v[0] for v in joint_limits.values()], device=actions.device)
        upper = torch.tensor([v[1] for v in joint_limits.values()], device=actions.device)
        actions = torch.clamp(actions, lower, upper)

        # Set joint targets
        self._robot.set_joint_position_target(actions, joint_ids=list(range(6)))

    def reset_robot_to_default(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Reset robot to default configuration."""
        if env_ids is None:
            env_ids = range(self.num_envs)

        default_pos = torch.tensor(
            list(UR5E_WITH_CAMERAS_CFG.robot.init_state.joint_pos.values()),
            device=self.device,
        ).unsqueeze(0).expand(len(env_ids), -1)

        default_vel = torch.zeros_like(default_pos)

        self._robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)

    def get_camera_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera images as numpy arrays for VLM processing.

        Returns:
            context_img: Context camera image [H, W, 3] uint8
            wrist_img: Wrist camera image [H, W, 3] uint8
        """
        context_cam: Camera = self.scene["context_camera"]
        wrist_cam: Camera = self.scene["wrist_camera"]

        context_rgb = context_cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        wrist_rgb = wrist_cam.data.output["rgb"][0, ..., :3].cpu().numpy()

        if context_rgb.dtype != np.uint8:
            context_rgb = (context_rgb * 255).astype(np.uint8)
            wrist_rgb = (wrist_rgb * 255).astype(np.uint8)

        return context_rgb, wrist_rgb


def create_ur5e_vla_env(
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = False,
) -> UR5eVLAEnv:
    """
    Factory function to create UR5e VLA environment.

    Args:
        num_envs: Number of parallel environments
        device: Compute device
        headless: Run without visualization

    Returns:
        Configured UR5eVLAEnv instance
    """
    cfg = UR5eVLAEnvCfg()
    cfg.scene.num_envs = num_envs

    render_mode = None if headless else "human"

    return UR5eVLAEnv(cfg, render_mode=render_mode)
