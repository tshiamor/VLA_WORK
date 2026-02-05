"""
Reaching Task

Simple reaching task for VLA training - robot must move end-effector to goal.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform


@configclass
class ReachTaskCfg:
    """Configuration for reaching task."""

    # Goal marker (visual only)
    goal_marker_cfg: sim_utils.SpawnerCfg = sim_utils.SphereCfg(
        radius=0.02,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0),  # Green sphere
            opacity=0.8,
        ),
    )

    # Task parameters
    goal_pos_nominal: Tuple[float, float, float] = (0.5, 0.0, 0.3)  # Nominal goal position
    goal_spawn_range: Tuple[float, float, float] = (0.2, 0.2, 0.15)  # Random range
    success_threshold: float = 0.02  # 2cm tolerance
    velocity_threshold: float = 0.05  # Must be nearly stationary

    # Reward weights
    reward_distance: float = 1.0
    reward_velocity: float = 0.1
    reward_success: float = 10.0

    # Language instructions
    instructions: List[str] = field(default_factory=lambda: [
        "Move the robot arm to the green target.",
        "Reach for the green sphere.",
        "Position the end-effector at the goal location.",
        "Navigate the arm to touch the green marker.",
        "Move to the indicated position.",
    ])


class ReachTask:
    """
    Simple reaching task.

    The robot must move its end-effector to a randomly sampled goal position.
    This is a foundational task for VLA training.
    """

    def __init__(self, cfg: ReachTaskCfg, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = device

        # Task state
        self._goal_positions: Optional[torch.Tensor] = None
        self._prev_dist: Optional[torch.Tensor] = None

    def setup(self, num_envs: int) -> None:
        """Initialize task state buffers."""
        self._goal_positions = torch.zeros(num_envs, 3, device=self.device)
        self._prev_dist = torch.ones(num_envs, device=self.device) * float('inf')

    def reset(self, env_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Reset task for specified environments.

        Args:
            env_ids: Environment indices to reset

        Returns:
            Dictionary with reset information including goal positions
        """
        num_resets = len(env_ids)

        # Sample random goal positions
        goal_pos = torch.zeros(num_resets, 3, device=self.device)

        # X: forward/backward from robot base
        goal_pos[:, 0] = self.cfg.goal_pos_nominal[0] + sample_uniform(
            -self.cfg.goal_spawn_range[0],
            self.cfg.goal_spawn_range[0],
            (num_resets,),
            device=self.device,
        )

        # Y: left/right
        goal_pos[:, 1] = self.cfg.goal_pos_nominal[1] + sample_uniform(
            -self.cfg.goal_spawn_range[1],
            self.cfg.goal_spawn_range[1],
            (num_resets,),
            device=self.device,
        )

        # Z: up/down (keep above table)
        goal_pos[:, 2] = self.cfg.goal_pos_nominal[2] + sample_uniform(
            -self.cfg.goal_spawn_range[2],
            self.cfg.goal_spawn_range[2],
            (num_resets,),
            device=self.device,
        )
        goal_pos[:, 2] = torch.clamp(goal_pos[:, 2], min=0.1)  # Minimum height

        # Store goal positions
        self._goal_positions[env_ids] = goal_pos

        # Reset distance tracking
        self._prev_dist[env_ids] = float('inf')

        return {
            "goal_pos": goal_pos,
        }

    def compute_reward(
        self,
        ee_pos: torch.Tensor,
        ee_vel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute task reward.

        Args:
            ee_pos: End-effector position [num_envs, 3]
            ee_vel: End-effector velocity [num_envs, 3] (optional)

        Returns:
            reward: Total reward [num_envs]
            info: Dictionary with reward components and metrics
        """
        # Distance to goal
        dist = torch.norm(ee_pos - self._goal_positions, dim=-1)

        # Distance-based reward (negative distance)
        distance_reward = -dist * self.cfg.reward_distance

        # Progress reward (getting closer)
        progress = self._prev_dist - dist
        progress_reward = torch.clamp(progress, min=0) * 5.0  # Bonus for progress
        self._prev_dist = dist.clone()

        # Velocity penalty (encourage smooth motion)
        velocity_penalty = torch.zeros_like(dist)
        if ee_vel is not None:
            vel_norm = torch.norm(ee_vel, dim=-1)
            velocity_penalty = -vel_norm * self.cfg.reward_velocity

        # Success bonus
        is_success = dist < self.cfg.success_threshold
        if ee_vel is not None:
            vel_norm = torch.norm(ee_vel, dim=-1)
            is_success = is_success & (vel_norm < self.cfg.velocity_threshold)

        success_reward = is_success.float() * self.cfg.reward_success

        # Total reward
        total_reward = distance_reward + progress_reward + velocity_penalty + success_reward

        info = {
            "distance_reward": distance_reward,
            "progress_reward": progress_reward,
            "velocity_penalty": velocity_penalty,
            "success_reward": success_reward,
            "distance": dist,
            "is_success": is_success,
        }

        return total_reward, info

    def check_termination(
        self,
        ee_pos: torch.Tensor,
        ee_vel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check for episode termination.

        Args:
            ee_pos: End-effector position
            ee_vel: End-effector velocity (optional)

        Returns:
            terminated: Whether episode ended due to success [num_envs]
            truncated: Whether episode was truncated [num_envs]
        """
        dist = torch.norm(ee_pos - self._goal_positions, dim=-1)

        # Success: reached goal and (optionally) stationary
        is_success = dist < self.cfg.success_threshold
        if ee_vel is not None:
            vel_norm = torch.norm(ee_vel, dim=-1)
            is_success = is_success & (vel_norm < self.cfg.velocity_threshold)

        terminated = is_success
        truncated = torch.zeros_like(terminated)

        return terminated, truncated

    def get_instruction(self, env_idx: int = 0) -> str:
        """Get a random language instruction for the task."""
        idx = np.random.randint(len(self.cfg.instructions))
        return self.cfg.instructions[idx]

    def get_goal_position(self, env_idx: int = 0) -> torch.Tensor:
        """Get goal position for specified environment."""
        return self._goal_positions[env_idx]

    def get_observations(self, env_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Get task-specific observations.

        Args:
            env_idx: Environment index

        Returns:
            Dictionary with goal position
        """
        return {
            "goal_pos": self._goal_positions[env_idx],
        }


class MultiGoalReachTask(ReachTask):
    """
    Extended reaching task with multiple sequential goals.

    The robot must reach a sequence of waypoints in order.
    """

    def __init__(
        self,
        cfg: ReachTaskCfg,
        device: str = "cuda:0",
        num_waypoints: int = 3,
    ):
        super().__init__(cfg, device)
        self.num_waypoints = num_waypoints
        self._current_waypoint: Optional[torch.Tensor] = None
        self._all_waypoints: Optional[torch.Tensor] = None

    def setup(self, num_envs: int) -> None:
        """Initialize task state buffers."""
        super().setup(num_envs)
        self._current_waypoint = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._all_waypoints = torch.zeros(
            num_envs, self.num_waypoints, 3, device=self.device
        )

    def reset(self, env_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Reset task with multiple waypoints."""
        num_resets = len(env_ids)

        # Generate all waypoints
        all_waypoints = torch.zeros(num_resets, self.num_waypoints, 3, device=self.device)

        for i in range(self.num_waypoints):
            wp = torch.zeros(num_resets, 3, device=self.device)
            wp[:, 0] = self.cfg.goal_pos_nominal[0] + sample_uniform(
                -self.cfg.goal_spawn_range[0],
                self.cfg.goal_spawn_range[0],
                (num_resets,),
                device=self.device,
            )
            wp[:, 1] = self.cfg.goal_pos_nominal[1] + sample_uniform(
                -self.cfg.goal_spawn_range[1],
                self.cfg.goal_spawn_range[1],
                (num_resets,),
                device=self.device,
            )
            wp[:, 2] = self.cfg.goal_pos_nominal[2] + sample_uniform(
                -self.cfg.goal_spawn_range[2],
                self.cfg.goal_spawn_range[2],
                (num_resets,),
                device=self.device,
            )
            wp[:, 2] = torch.clamp(wp[:, 2], min=0.1)
            all_waypoints[:, i] = wp

        self._all_waypoints[env_ids] = all_waypoints
        self._current_waypoint[env_ids] = 0

        # Set first waypoint as current goal
        self._goal_positions[env_ids] = all_waypoints[:, 0]
        self._prev_dist[env_ids] = float('inf')

        return {
            "all_waypoints": all_waypoints,
            "goal_pos": all_waypoints[:, 0],
        }

    def advance_waypoint(self, env_ids: torch.Tensor) -> None:
        """Advance to next waypoint for specified environments."""
        self._current_waypoint[env_ids] += 1

        # Update goal positions for envs that haven't finished all waypoints
        for env_id in env_ids:
            wp_idx = self._current_waypoint[env_id].item()
            if wp_idx < self.num_waypoints:
                self._goal_positions[env_id] = self._all_waypoints[env_id, wp_idx]
                self._prev_dist[env_id] = float('inf')

    def check_termination(
        self,
        ee_pos: torch.Tensor,
        ee_vel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check termination, advancing waypoints as needed."""
        dist = torch.norm(ee_pos - self._goal_positions, dim=-1)

        # Check if current waypoint reached
        reached = dist < self.cfg.success_threshold
        if ee_vel is not None:
            vel_norm = torch.norm(ee_vel, dim=-1)
            reached = reached & (vel_norm < self.cfg.velocity_threshold)

        # Advance waypoint for those who reached
        reached_ids = torch.where(reached)[0]
        if len(reached_ids) > 0:
            self.advance_waypoint(reached_ids)

        # Terminated if all waypoints completed
        all_done = self._current_waypoint >= self.num_waypoints
        terminated = all_done
        truncated = torch.zeros_like(terminated)

        return terminated, truncated
