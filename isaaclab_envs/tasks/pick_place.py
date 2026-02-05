"""
Pick and Place Task

Defines pick-and-place manipulation task for VLA training and evaluation.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz, sample_uniform


@configclass
class PickPlaceTaskCfg:
    """Configuration for pick-and-place task."""

    # Object to manipulate
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red cube
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.025),  # On table surface
        ),
    )

    # Goal marker (visual only)
    goal_marker_cfg: sim_utils.SpawnerCfg = sim_utils.CylinderCfg(
        radius=0.03,
        height=0.002,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0),  # Green marker
            opacity=0.5,
        ),
    )

    # Task parameters
    object_spawn_range: Tuple[float, float, float] = (0.1, 0.1, 0.0)  # xyz range around nominal
    goal_spawn_range: Tuple[float, float, float] = (0.15, 0.15, 0.0)
    success_threshold: float = 0.03  # 3cm tolerance
    lift_height: float = 0.15  # Required lift height

    # Reward weights
    reward_reaching: float = 1.0
    reward_grasping: float = 2.0
    reward_lifting: float = 2.0
    reward_placing: float = 5.0
    reward_success: float = 10.0

    # Language instructions
    instructions: List[str] = field(default_factory=lambda: [
        "Pick up the red cube and place it on the green marker.",
        "Grasp the cube and move it to the goal location.",
        "Move the red object to the green target area.",
        "Pick the cube and put it down at the marked position.",
    ])


class PickPlaceTask:
    """
    Pick and place manipulation task.

    The robot must:
    1. Reach and grasp an object
    2. Lift it above the table
    3. Move it to a goal location
    4. Place it down gently
    """

    def __init__(self, cfg: PickPlaceTaskCfg, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = device

        # Task state tracking
        self._object: Optional[RigidObject] = None
        self._goal_positions: Optional[torch.Tensor] = None
        self._object_grasped: Optional[torch.Tensor] = None
        self._object_lifted: Optional[torch.Tensor] = None

    def setup(self, scene) -> None:
        """Setup task objects in the scene."""
        # Add manipulated object
        self._object = scene.add(self.cfg.object_cfg)

        # Initialize state tracking
        num_envs = scene.num_envs
        self._goal_positions = torch.zeros(num_envs, 3, device=self.device)
        self._object_grasped = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._object_lifted = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Reset task for specified environments.

        Args:
            env_ids: Environment indices to reset

        Returns:
            Dictionary with reset information
        """
        num_resets = len(env_ids)

        # Randomize object position
        obj_pos = torch.zeros(num_resets, 3, device=self.device)
        obj_pos[:, 0] = 0.5 + sample_uniform(
            -self.cfg.object_spawn_range[0],
            self.cfg.object_spawn_range[0],
            (num_resets,),
            device=self.device,
        )
        obj_pos[:, 1] = sample_uniform(
            -self.cfg.object_spawn_range[1],
            self.cfg.object_spawn_range[1],
            (num_resets,),
            device=self.device,
        )
        obj_pos[:, 2] = 0.025  # On table

        # Set object state
        obj_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num_resets, -1)
        obj_vel = torch.zeros(num_resets, 6, device=self.device)

        self._object.write_root_pose_to_sim(
            torch.cat([obj_pos, obj_quat], dim=-1), env_ids=env_ids
        )
        self._object.write_root_velocity_to_sim(obj_vel, env_ids=env_ids)

        # Randomize goal position (different from object)
        goal_pos = torch.zeros(num_resets, 3, device=self.device)
        goal_pos[:, 0] = 0.4 + sample_uniform(
            -self.cfg.goal_spawn_range[0],
            self.cfg.goal_spawn_range[0],
            (num_resets,),
            device=self.device,
        )
        goal_pos[:, 1] = 0.2 + sample_uniform(
            -self.cfg.goal_spawn_range[1],
            self.cfg.goal_spawn_range[1],
            (num_resets,),
            device=self.device,
        )
        goal_pos[:, 2] = 0.001  # On table surface

        self._goal_positions[env_ids] = goal_pos

        # Reset state tracking
        self._object_grasped[env_ids] = False
        self._object_lifted[env_ids] = False

        return {
            "object_pos": obj_pos,
            "goal_pos": goal_pos,
        }

    def compute_reward(
        self,
        ee_pos: torch.Tensor,
        gripper_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute task reward.

        Args:
            ee_pos: End-effector position [num_envs, 3]
            gripper_state: Gripper open/close state [num_envs]

        Returns:
            reward: Total reward [num_envs]
            info: Dictionary with reward components
        """
        num_envs = ee_pos.shape[0]

        # Get object position
        obj_pos = self._object.data.root_pos_w[:, :3]

        # Distance: EE to object
        ee_to_obj_dist = torch.norm(ee_pos - obj_pos, dim=-1)

        # Distance: object to goal
        obj_to_goal_dist = torch.norm(obj_pos[:, :2] - self._goal_positions[:, :2], dim=-1)

        # Reaching reward (approach object)
        reaching_reward = -ee_to_obj_dist * self.cfg.reward_reaching

        # Grasping reward (object lifted with gripper closed)
        is_grasping = (ee_to_obj_dist < 0.05) & (gripper_state < 0.02)
        grasping_reward = is_grasping.float() * self.cfg.reward_grasping
        self._object_grasped = self._object_grasped | is_grasping

        # Lifting reward
        is_lifted = obj_pos[:, 2] > self.cfg.lift_height
        lifting_reward = is_lifted.float() * self.cfg.reward_lifting
        self._object_lifted = self._object_lifted | is_lifted

        # Placing reward (object at goal)
        placing_reward = -obj_to_goal_dist * self.cfg.reward_placing

        # Success bonus
        is_success = obj_to_goal_dist < self.cfg.success_threshold
        is_placed = is_success & (obj_pos[:, 2] < 0.05)  # On table at goal
        success_reward = is_placed.float() * self.cfg.reward_success

        # Total reward
        total_reward = reaching_reward + grasping_reward + lifting_reward + placing_reward + success_reward

        info = {
            "reaching_reward": reaching_reward,
            "grasping_reward": grasping_reward,
            "lifting_reward": lifting_reward,
            "placing_reward": placing_reward,
            "success_reward": success_reward,
            "ee_to_obj_dist": ee_to_obj_dist,
            "obj_to_goal_dist": obj_to_goal_dist,
            "is_success": is_placed,
        }

        return total_reward, info

    def check_termination(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check for episode termination conditions.

        Returns:
            terminated: Whether episode ended due to success/failure [num_envs]
            truncated: Whether episode was truncated (time limit) [num_envs]
        """
        obj_pos = self._object.data.root_pos_w[:, :3]
        obj_to_goal_dist = torch.norm(obj_pos[:, :2] - self._goal_positions[:, :2], dim=-1)

        # Success: object at goal on table
        success = (obj_to_goal_dist < self.cfg.success_threshold) & (obj_pos[:, 2] < 0.05)

        # Failure: object fell off table
        fell_off = obj_pos[:, 2] < -0.1

        terminated = success | fell_off
        truncated = torch.zeros_like(terminated)  # Time-based truncation handled by env

        return terminated, truncated

    def get_instruction(self, env_idx: int = 0) -> str:
        """Get a random language instruction for the task."""
        idx = np.random.randint(len(self.cfg.instructions))
        return self.cfg.instructions[idx]

    def get_goal_position(self, env_idx: int = 0) -> torch.Tensor:
        """Get goal position for specified environment."""
        return self._goal_positions[env_idx]

    def get_object_position(self, env_idx: int = 0) -> torch.Tensor:
        """Get object position for specified environment."""
        return self._object.data.root_pos_w[env_idx, :3]
