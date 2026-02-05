#!/usr/bin/env python3
"""
VLA Evaluation Script

Evaluate a trained VLA model in Isaac Lab simulation.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --robot franka
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --robot ur5e --task reach
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

from vla.models import VLAModel, create_vla_model
from vla.utils.transforms import ActionChunker


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLA model")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Robot and task
    parser.add_argument("--robot", type=str, default="franka",
                        choices=["franka", "ur5e"],
                        help="Robot type")
    parser.add_argument("--task", type=str, default="reach",
                        choices=["reach", "pick_place"],
                        help="Task to evaluate")

    # Evaluation settings
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum steps per episode")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override instruction (uses task default if not specified)")

    # Visualization
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization")
    parser.add_argument("--save_video", action="store_true",
                        help="Save evaluation videos")
    parser.add_argument("--video_dir", type=str, default="videos",
                        help="Directory to save videos")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    return parser.parse_args()


class VLAEvaluator:
    """Evaluator for VLA models in simulation."""

    def __init__(
        self,
        model: VLAModel,
        robot_type: str,
        task_type: str,
        device: str = "cuda",
    ):
        self.model = model
        self.robot_type = robot_type
        self.task_type = task_type
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Action chunker
        self.chunker = ActionChunker(
            chunk_size=model.config.chunk_size,
            overlap=4,  # Use some overlap for smoothness
        )

        # Environment and task (lazy initialization)
        self._env = None
        self._task = None

    def _init_environment(self, headless: bool = False):
        """Initialize Isaac Lab environment."""
        # Import here to avoid loading Isaac Lab unless needed
        if self.robot_type == "franka":
            from isaaclab_envs.envs import FrankaVLAEnv, create_franka_vla_env
            self._env = create_franka_vla_env(num_envs=1, headless=headless)
        else:
            from isaaclab_envs.envs import UR5eVLAEnv, create_ur5e_vla_env
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

    @torch.no_grad()
    def evaluate_episode(
        self,
        instruction: Optional[str] = None,
        max_steps: int = 200,
    ) -> dict:
        """
        Run a single evaluation episode.

        Returns:
            Episode metrics (success, reward, steps)
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call _init_environment first.")

        # Reset environment and task
        self._env.reset()
        task_info = self._task.reset(torch.tensor([0], device=self.device))
        self.chunker.reset()

        # Get instruction
        if instruction is None:
            instruction = self._task.get_instruction()

        # Episode tracking
        episode_reward = 0.0
        episode_steps = 0
        success = False

        for step in range(max_steps):
            # Get observations
            obs = self._env.get_observations()
            context_img, wrist_img = self._env.get_camera_images()
            proprio = obs['proprio']

            # Predict actions
            actions = self.model.predict_action(
                images=[context_img, wrist_img],
                instruction=instruction,
                proprio_state=proprio,
            )

            # Get current action from chunk
            action = self.chunker.update(actions[0].cpu().numpy())

            # Execute action
            self._env.apply_action(torch.tensor(action, device=self.device).unsqueeze(0))
            self._env.step()

            # Get EE position for task evaluation
            ee_pos = obs.get('ee_position', proprio[:, :3])

            # Compute reward
            reward, info = self._task.compute_reward(ee_pos)
            episode_reward += reward.item()

            # Check termination
            terminated, truncated = self._task.check_termination(ee_pos)
            if terminated.any():
                success = info.get('is_success', terminated).item()
                break

            episode_steps += 1

        return {
            'success': success,
            'reward': episode_reward,
            'steps': episode_steps,
        }

    def evaluate(
        self,
        num_episodes: int = 50,
        instruction: Optional[str] = None,
        max_steps: int = 200,
        headless: bool = False,
    ) -> dict:
        """
        Run full evaluation.

        Returns:
            Aggregated metrics across all episodes
        """
        # Initialize environment
        print("Initializing environment...")
        self._init_environment(headless=headless)

        # Run episodes
        successes = []
        rewards = []
        steps_list = []

        print(f"\nRunning {num_episodes} evaluation episodes...")
        for ep in range(num_episodes):
            result = self.evaluate_episode(instruction, max_steps)
            successes.append(result['success'])
            rewards.append(result['reward'])
            steps_list.append(result['steps'])

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{num_episodes} - "
                      f"Success Rate: {np.mean(successes)*100:.1f}%")

        # Compute final metrics
        metrics = {
            'success_rate': np.mean(successes),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps_list),
            'num_episodes': num_episodes,
        }

        return metrics


def load_model(checkpoint_path: str, device: str = "cuda") -> VLAModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config from checkpoint
    config = checkpoint.get('config', {})

    # Create model
    model = create_vla_model(
        action_dim=config.get('action_dim', 7),
        chunk_size=config.get('chunk_size', 16),
        hidden_dim=config.get('hidden_dim', 512),
        device_map='cpu',  # Load to CPU first
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return model


def main():
    args = parse_args()

    print("=" * 60)
    print("VLA Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Robot: {args.robot}")
    print(f"Task: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, args.device)
    print("Model loaded successfully!")

    # Create evaluator
    evaluator = VLAEvaluator(
        model=model,
        robot_type=args.robot,
        task_type=args.task,
        device=args.device,
    )

    # Run evaluation
    metrics = evaluator.evaluate(
        num_episodes=args.num_episodes,
        instruction=args.instruction,
        max_steps=args.max_steps,
        headless=args.headless,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} (+/- {metrics['std_reward']:.2f})")
    print(f"Mean Steps: {metrics['mean_steps']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
