#!/usr/bin/env python3
"""
Run VLA Model in Isaac Lab Lift Environment

This script demonstrates running a VLA model for the cube lifting task.
It can run in demo mode (scripted policy) or with a trained VLA model.

Usage:
    # Demo mode with scripted policy
    python run_vla_lift.py --mode demo --num_episodes 5

    # With trained VLA model
    python run_vla_lift.py --mode vla --checkpoint /path/to/checkpoint.pt

    # Collect demonstrations
    python run_vla_lift.py --mode collect --output_dir ./demos --num_episodes 100
"""

import argparse
import sys
import os
from pathlib import Path

# Add paths
VLA_WORK_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(VLA_WORK_PATH))
sys.path.insert(0, str(VLA_WORK_PATH / "isaaclab_ext"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run VLA in Isaac Lab Lift Environment")

    parser.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "vla", "collect"],
                        help="Run mode: demo (scripted), vla (model), collect (demos)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to VLA model checkpoint")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--output_dir", type=str, default="./demos",
                        help="Output directory for collected demonstrations")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for VLA model")

    return parser.parse_args()


def run_demo_mode(args):
    """Run with scripted reaching policy."""
    import torch
    import gymnasium as gym

    # Import VLA tasks to register environments
    import ext_vla_tasks

    print("Creating VLA-Franka-Lift environment...")
    env = gym.make(
        "VLA-Franka-Lift-Play-v0",
        render_mode=None if args.headless else "human",
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for episode in range(args.num_episodes):
        print(f"\n=== Episode {episode + 1}/{args.num_episodes} ===")
        obs, info = env.reset()

        done = False
        step = 0
        total_reward = 0

        while not done:
            # Simple scripted policy: move toward object
            # In real use, this would be replaced by VLA model
            action = env.action_space.sample() * 0.1  # Small random actions

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step += 1

            # Print camera observation shapes
            if step == 1:
                if "camera" in obs:
                    print(f"Camera observations available:")
                    for key, val in obs["camera"].items():
                        print(f"  {key}: {val.shape}")

        print(f"Episode finished: steps={step}, reward={total_reward:.2f}")

    env.close()
    print("\nDemo complete!")


def run_vla_mode(args):
    """Run with trained VLA model."""
    import torch
    import gymnasium as gym
    import numpy as np

    # Import VLA components
    from vla.models import create_vla_model
    from vla.utils.transforms import ActionChunker

    # Import VLA tasks
    import ext_vla_tasks

    if args.checkpoint is None:
        print("Error: --checkpoint required for VLA mode")
        return

    print("Loading VLA model...")
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})

    model = create_vla_model(
        action_dim=config.get("action_dim", 8),  # 7 arm + 1 gripper
        chunk_size=config.get("chunk_size", 16),
        hidden_dim=config.get("hidden_dim", 512),
        device_map="cpu",
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(args.device)
    model.eval()

    print("Creating environment...")
    env = gym.make(
        "VLA-Franka-Lift-Play-v0",
        render_mode=None if args.headless else "human",
    )

    # Action chunker for temporal consistency
    chunker = ActionChunker(chunk_size=16, overlap=4)

    # Language instruction
    instruction = "Pick up the red cube and lift it above the table."

    successes = 0
    for episode in range(args.num_episodes):
        print(f"\n=== Episode {episode + 1}/{args.num_episodes} ===")
        obs, info = env.reset()
        chunker.reset()

        done = False
        step = 0

        while not done:
            # Get camera images
            context_rgb = obs["camera"]["context_rgb"][0].cpu().numpy()
            wrist_rgb = obs["camera"]["wrist_rgb"][0].cpu().numpy()

            # Get proprioception
            proprio = obs["policy"][0].cpu().numpy()

            # Predict actions with VLA model
            with torch.no_grad():
                actions = model.predict_action(
                    images=[context_rgb, wrist_rgb],
                    instruction=instruction,
                    proprio_state=torch.tensor(proprio[:14], device=args.device).unsqueeze(0),
                )

            # Get current action from chunk
            action = chunker.update(actions[0].cpu().numpy())

            # Execute
            obs, reward, terminated, truncated, info = env.step(
                torch.tensor(action, device=env.device).unsqueeze(0)
            )
            done = terminated or truncated
            step += 1

        # Check success
        obj_height = env.scene["object"].data.root_pos_w[0, 2].item()
        success = obj_height > 0.15
        if success:
            successes += 1
        print(f"Episode finished: steps={step}, success={success}")

    print(f"\nSuccess rate: {successes}/{args.num_episodes} ({100*successes/args.num_episodes:.1f}%)")
    env.close()


def run_collect_mode(args):
    """Collect demonstrations using scripted policy."""
    import torch
    import gymnasium as gym
    import numpy as np
    import h5py
    from pathlib import Path

    # Import VLA tasks
    import ext_vla_tasks

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating environment for data collection...")
    env = gym.make(
        "VLA-Franka-Lift-Play-v0",
        render_mode=None if args.headless else "human",
    )

    instructions = [
        "Pick up the red cube and lift it.",
        "Grasp the cube and raise it above the table.",
        "Lift the red block.",
        "Pick up the object in front of you.",
    ]

    episode_count = 0
    while episode_count < args.num_episodes:
        print(f"\n=== Collecting Episode {episode_count + 1}/{args.num_episodes} ===")

        obs, info = env.reset()

        # Episode data buffers
        context_images = []
        wrist_images = []
        actions_list = []
        proprio_states = []

        instruction = instructions[episode_count % len(instructions)]

        done = False
        step = 0
        success = False

        while not done and step < 300:
            # Store observations
            context_rgb = obs["camera"]["context_rgb"][0].cpu().numpy()
            wrist_rgb = obs["camera"]["wrist_rgb"][0].cpu().numpy()
            proprio = obs["policy"][0].cpu().numpy()

            context_images.append((context_rgb * 255).astype(np.uint8))
            wrist_images.append((wrist_rgb * 255).astype(np.uint8))
            proprio_states.append(proprio)

            # Simple scripted policy for demo collection
            # Reach toward object, then close gripper, then lift
            obj_pos = env.scene["object"].data.root_pos_w[0].cpu().numpy()
            ee_pos = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()

            # Compute desired action
            direction = obj_pos - ee_pos
            distance = np.linalg.norm(direction)

            # Action: [7 joint deltas, 1 gripper]
            action = np.zeros(8)

            if distance > 0.05:
                # Reach phase
                action[:3] = direction * 2.0  # Simple proportional control
                action[7] = 1.0  # Open gripper
            elif obj_pos[2] < 0.1:
                # Grasp phase
                action[7] = -1.0  # Close gripper
            else:
                # Lift phase
                action[2] = 0.5  # Move up
                action[7] = -1.0  # Keep gripper closed

            # Clip actions
            action = np.clip(action, -1.0, 1.0)
            actions_list.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(
                torch.tensor(action, device=env.device).unsqueeze(0)
            )
            done = terminated or truncated
            step += 1

            # Check success
            obj_height = env.scene["object"].data.root_pos_w[0, 2].item()
            if obj_height > 0.15:
                success = True

        # Save successful episodes
        if success and len(actions_list) > 16:
            episode_path = output_dir / f"episode_{episode_count:06d}.h5"

            with h5py.File(episode_path, 'w') as f:
                f.create_dataset("context_images", data=np.stack(context_images), compression="gzip")
                f.create_dataset("wrist_images", data=np.stack(wrist_images), compression="gzip")
                f.create_dataset("actions", data=np.stack(actions_list))
                f.create_dataset("proprio_states", data=np.stack(proprio_states))
                f.attrs["instruction"] = instruction
                f.attrs["success"] = success
                f.attrs["num_steps"] = len(actions_list)

            print(f"Saved episode to {episode_path} (steps={len(actions_list)})")
            episode_count += 1
        else:
            print(f"Episode failed or too short, retrying...")

    env.close()
    print(f"\nCollected {episode_count} demonstrations to {output_dir}")


def main():
    args = parse_args()

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    if args.mode == "demo":
        run_demo_mode(args)
    elif args.mode == "vla":
        run_vla_mode(args)
    elif args.mode == "collect":
        run_collect_mode(args)

    simulation_app.close()


if __name__ == "__main__":
    main()
