#!/usr/bin/env python3
"""
Demonstration Collection Script

Collect robot demonstrations in Isaac Lab for VLA training.
Supports teleoperation and scripted policies.

Usage:
    python scripts/collect_demos.py --robot franka --task reach --num_episodes 100
    python scripts/collect_demos.py --robot ur5e --task pick_place --mode teleop
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vla.data.dataset import VLADemoDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Collect robot demonstrations")

    # Robot and task
    parser.add_argument("--robot", type=str, default="franka",
                        choices=["franka", "ur5e"],
                        help="Robot type")
    parser.add_argument("--task", type=str, default="reach",
                        choices=["reach", "pick_place"],
                        help="Task type")

    # Collection settings
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum steps per episode")
    parser.add_argument("--mode", type=str, default="scripted",
                        choices=["scripted", "teleop"],
                        help="Collection mode")

    # Output
    parser.add_argument("--output_dir", type=str, default="data/demos",
                        help="Directory to save demonstrations")

    # Visualization
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization")

    return parser.parse_args()


class ScriptedPolicy:
    """Scripted policy for demonstration collection."""

    def __init__(self, robot_type: str, task_type: str, device: str = "cuda"):
        self.robot_type = robot_type
        self.task_type = task_type
        self.device = torch.device(device)

        # Control gains
        self.kp = 5.0  # Proportional gain
        self.kd = 0.5  # Derivative gain

        self._target = None
        self._prev_error = None

    def set_target(self, target_pos: np.ndarray):
        """Set target position for reaching."""
        self._target = target_pos
        self._prev_error = None

    def get_action(
        self,
        ee_pos: np.ndarray,
        ee_vel: np.ndarray,
        joint_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Compute action to reach target.

        Uses simple Cartesian PD control with pseudo-inverse Jacobian.
        """
        if self._target is None:
            return np.zeros(joint_pos.shape[-1])

        # Cartesian error
        error = self._target - ee_pos

        if self._prev_error is None:
            self._prev_error = error

        # PD control in Cartesian space
        error_d = error - self._prev_error
        self._prev_error = error

        cartesian_vel = self.kp * error + self.kd * error_d

        # Simple mapping to joint space (placeholder - actual implementation
        # would use Jacobian)
        # For now, just return small joint deltas proportional to error
        action_dim = joint_pos.shape[-1]
        joint_action = np.zeros(action_dim)

        # Map x, y, z errors to relevant joints
        if action_dim >= 3:
            joint_action[:3] = cartesian_vel * 0.1

        # Return target joint positions
        return joint_pos + joint_action


class DemoCollector:
    """Demonstration collector for VLA training data."""

    def __init__(
        self,
        robot_type: str,
        task_type: str,
        output_dir: str,
        device: str = "cuda",
    ):
        self.robot_type = robot_type
        self.task_type = task_type
        self.device = torch.device(device)

        # Dataset for recording
        self.dataset = VLADemoDataset(
            output_dir,
            action_dim=7 if robot_type == "franka" else 6,
            proprio_dim=14 if robot_type == "franka" else 12,
        )

        # Environment and task
        self._env = None
        self._task = None

        # Scripted policy
        self._policy = ScriptedPolicy(robot_type, task_type, device)

    def _init_environment(self, headless: bool = False):
        """Initialize Isaac Lab environment."""
        if self.robot_type == "franka":
            from isaaclab_envs.envs import create_franka_vla_env
            self._env = create_franka_vla_env(num_envs=1, headless=headless)
        else:
            from isaaclab_envs.envs import create_ur5e_vla_env
            self._env = create_ur5e_vla_env(num_envs=1, headless=headless)

        # Initialize task
        if self.task_type == "reach":
            from isaaclab_envs.tasks import ReachTask, ReachTaskCfg
            self._task = ReachTask(ReachTaskCfg(), device=str(self.device))
            self._task.setup(1)
        else:
            from isaaclab_envs.tasks import PickPlaceTask, PickPlaceTaskCfg
            self._task = PickPlaceTask(PickPlaceTaskCfg(), device=str(self.device))
            self._task.setup(self._env.scene)

    def collect_episode_scripted(
        self,
        max_steps: int = 200,
    ) -> dict:
        """Collect one episode using scripted policy."""
        if self._env is None:
            raise RuntimeError("Environment not initialized")

        # Reset
        self._env.reset()
        task_info = self._task.reset(torch.tensor([0], device=self.device))

        # Get instruction and goal
        instruction = self._task.get_instruction()
        goal_pos = self._task.get_goal_position().cpu().numpy()

        # Set policy target
        self._policy.set_target(goal_pos)

        # Start recording
        self.dataset.start_recording(instruction)

        success = False
        for step in range(max_steps):
            # Get observations
            obs = self._env.get_observations()
            context_img, wrist_img = self._env.get_camera_images()

            proprio = obs['proprio'][0].cpu().numpy()
            joint_pos = proprio[:7] if self.robot_type == "franka" else proprio[:6]
            joint_vel = proprio[7:] if self.robot_type == "franka" else proprio[6:]

            # Get EE position (simplified - would come from FK in real impl)
            ee_pos = obs.get('ee_position', torch.zeros(1, 3))[0].cpu().numpy()
            ee_vel = np.zeros(3)

            # Get action from policy
            action = self._policy.get_action(ee_pos, ee_vel, joint_pos)

            # Record step
            self.dataset.record_step(
                context_img=context_img,
                wrist_img=wrist_img,
                action=action,
                proprio_state=proprio,
            )

            # Execute action
            self._env.apply_action(torch.tensor(action, device=self.device).unsqueeze(0))
            self._env.step()

            # Check success
            ee_pos_tensor = torch.tensor(ee_pos, device=self.device).unsqueeze(0)
            terminated, _ = self._task.check_termination(ee_pos_tensor)
            if terminated.any():
                _, info = self._task.compute_reward(ee_pos_tensor)
                success = info.get('is_success', terminated).item()
                break

        # Stop recording
        episode = self.dataset.stop_recording(save=success)

        return {
            'success': success,
            'steps': step + 1,
            'saved': episode is not None,
        }

    def collect(
        self,
        num_episodes: int,
        max_steps: int = 200,
        mode: str = "scripted",
        headless: bool = False,
    ) -> dict:
        """Collect multiple episodes."""
        print(f"Initializing environment ({self.robot_type})...")
        self._init_environment(headless=headless)

        successful = 0
        total_attempts = 0

        print(f"\nCollecting {num_episodes} demonstrations...")
        while successful < num_episodes:
            total_attempts += 1

            if mode == "scripted":
                result = self.collect_episode_scripted(max_steps)
            else:
                # Teleoperation mode would require additional input handling
                raise NotImplementedError("Teleop mode not yet implemented")

            if result['saved']:
                successful += 1
                print(f"Episode {successful}/{num_episodes} saved "
                      f"(steps: {result['steps']}, attempts: {total_attempts})")

        print(f"\nCollection complete!")
        print(f"Saved {successful} episodes from {total_attempts} attempts")
        print(f"Success rate: {100*successful/total_attempts:.1f}%")

        return {
            'num_episodes': successful,
            'total_attempts': total_attempts,
            'success_rate': successful / total_attempts,
        }


def main():
    args = parse_args()

    print("=" * 60)
    print("Demonstration Collection")
    print("=" * 60)
    print(f"Robot: {args.robot}")
    print(f"Task: {args.task}")
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Create collector
    collector = DemoCollector(
        robot_type=args.robot,
        task_type=args.task,
        output_dir=args.output_dir,
    )

    # Collect demonstrations
    stats = collector.collect(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        mode=args.mode,
        headless=args.headless,
    )

    print(f"\nDemonstrations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
