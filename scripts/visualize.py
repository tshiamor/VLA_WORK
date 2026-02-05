#!/usr/bin/env python3
"""
VLA Visualization Script

Visualize VLA model predictions and attention maps.

Usage:
    python scripts/visualize.py --checkpoint checkpoints/best.pt --mode attention
    python scripts/visualize.py --checkpoint checkpoints/best.pt --mode actions
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional, List
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vla.models import VLAModel, create_vla_model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VLA model")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="actions",
                        choices=["actions", "attention", "flow"],
                        help="Visualization mode")

    # Input
    parser.add_argument("--context_image", type=str, default=None,
                        help="Path to context camera image")
    parser.add_argument("--wrist_image", type=str, default=None,
                        help="Path to wrist camera image")
    parser.add_argument("--instruction", type=str,
                        default="Move the robot arm to the green target.",
                        help="Language instruction")

    # Output
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")

    # Sampling
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of action samples to visualize")
    parser.add_argument("--num_flow_steps", type=int, default=20,
                        help="Number of flow steps to visualize")

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def load_model(checkpoint_path: str) -> VLAModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    model = create_vla_model(
        action_dim=config.get('action_dim', 7),
        chunk_size=config.get('chunk_size', 16),
        hidden_dim=config.get('hidden_dim', 512),
        device_map='cpu',
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model


def create_dummy_images() -> tuple:
    """Create dummy images for visualization without real data."""
    # Random colored images
    context_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    wrist_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Add some structure
    context_img[50:150, 50:150] = [255, 0, 0]  # Red square
    context_img[100:120, 160:180] = [0, 255, 0]  # Green target
    wrist_img[80:140, 80:140] = [100, 100, 200]  # Blue object

    return context_img, wrist_img


def visualize_actions(
    model: VLAModel,
    context_img: np.ndarray,
    wrist_img: np.ndarray,
    instruction: str,
    num_samples: int,
    output_dir: Path,
    device: str,
):
    """Visualize sampled action distributions."""
    model.to(device)
    model.eval()

    # Sample multiple action trajectories
    actions_list = []
    for _ in range(num_samples):
        with torch.no_grad():
            actions = model.predict_action(
                images=[context_img, wrist_img],
                instruction=instruction,
            )
        actions_list.append(actions[0].cpu().numpy())

    actions_array = np.stack(actions_list)  # [num_samples, chunk_size, action_dim]

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Sampled Action Distributions\n"{instruction}"', fontsize=12)

    chunk_size, action_dim = actions_array.shape[1], actions_array.shape[2]
    timesteps = np.arange(chunk_size)

    for i in range(min(action_dim, 7)):
        ax = axes[i // 4, i % 4]

        # Plot all samples
        for sample in actions_array:
            ax.plot(timesteps, sample[:, i], alpha=0.3, color='blue')

        # Plot mean and std
        mean = actions_array[:, :, i].mean(axis=0)
        std = actions_array[:, :, i].std(axis=0)
        ax.plot(timesteps, mean, color='red', linewidth=2, label='Mean')
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.3, color='red')

        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'Joint {i+1}')
        ax.set_title(f'Action Dimension {i+1}')
        ax.legend()

    # Hide unused subplots
    for i in range(action_dim, 8):
        axes[i // 4, i % 4].axis('off')

    plt.tight_layout()
    output_path = output_dir / 'action_distribution.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved action visualization to {output_path}")

    # Also save input images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(context_img)
    axes[0].set_title('Context Camera')
    axes[0].axis('off')
    axes[1].imshow(wrist_img)
    axes[1].set_title('Wrist Camera')
    axes[1].axis('off')
    plt.suptitle(f'Input Images\n"{instruction}"')
    plt.tight_layout()
    plt.savefig(output_dir / 'input_images.png', dpi=150)
    plt.close()


def visualize_flow(
    model: VLAModel,
    context_img: np.ndarray,
    wrist_img: np.ndarray,
    instruction: str,
    num_steps: int,
    output_dir: Path,
    device: str,
):
    """Visualize flow matching trajectory from noise to actions."""
    model.to(device)
    model.eval()

    # Initialize flow head for step-by-step visualization
    flow_head = model.flow_head
    chunk_size = model.config.chunk_size
    action_dim = model.config.action_dim

    # Encode observation to get conditioning
    with torch.no_grad():
        encoded = model.encode_observation(
            images=[context_img, wrist_img],
            instruction=instruction,
        )
        condition = encoded['condition']

    # Track trajectory through flow
    x_t = torch.randn(1, chunk_size * action_dim, device=device)
    trajectory = [x_t.cpu().numpy().reshape(chunk_size, action_dim)]

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((1,), i / num_steps, device=device)
        with torch.no_grad():
            v = flow_head.forward(x_t, t, condition)
        x_t = x_t + v * dt
        trajectory.append(x_t.cpu().numpy().reshape(chunk_size, action_dim))

    trajectory = np.stack(trajectory)  # [num_steps+1, chunk_size, action_dim]

    # Plot flow trajectory for first timestep of chunk
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Flow Matching Trajectory\n"{instruction}"', fontsize=12)

    t_values = np.linspace(0, 1, num_steps + 1)

    for i in range(min(action_dim, 7)):
        ax = axes[i // 4, i % 4]

        # Plot trajectory for first timestep in chunk
        ax.plot(t_values, trajectory[:, 0, i], linewidth=2)
        ax.scatter([0], [trajectory[0, 0, i]], color='blue', s=100, label='Noise', zorder=5)
        ax.scatter([1], [trajectory[-1, 0, i]], color='red', s=100, label='Action', zorder=5)

        ax.set_xlabel('Flow time t')
        ax.set_ylabel(f'Action value')
        ax.set_title(f'Dimension {i+1}')
        ax.legend()

    for i in range(action_dim, 8):
        axes[i // 4, i % 4].axis('off')

    plt.tight_layout()
    output_path = output_dir / 'flow_trajectory.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved flow visualization to {output_path}")


def visualize_attention(
    model: VLAModel,
    context_img: np.ndarray,
    wrist_img: np.ndarray,
    instruction: str,
    output_dir: Path,
    device: str,
):
    """Visualize attention maps from action expert."""
    model.to(device)
    model.eval()

    print("Note: Attention visualization requires model to return attention weights.")
    print("This is a placeholder - full implementation would extract attention from transformer.")

    # Placeholder visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(context_img)
    axes[0].set_title('Context Image')
    axes[0].axis('off')

    axes[1].imshow(wrist_img)
    axes[1].set_title('Wrist Image')
    axes[1].axis('off')

    # Dummy attention heatmap
    attention = np.random.rand(224, 224)
    attention[50:150, 50:150] += 0.5  # Higher attention on object
    attention = np.clip(attention, 0, 1)

    axes[2].imshow(context_img)
    axes[2].imshow(attention, alpha=0.5, cmap='jet')
    axes[2].set_title('Attention Overlay (Placeholder)')
    axes[2].axis('off')

    plt.suptitle(f'Attention Visualization\n"{instruction}"')
    plt.tight_layout()
    output_path = output_dir / 'attention.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved attention visualization to {output_path}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VLA Visualization")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Instruction: {args.instruction}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint)
    print("Model loaded!")

    # Load or create images
    if args.context_image and args.wrist_image:
        context_img = np.array(Image.open(args.context_image))
        wrist_img = np.array(Image.open(args.wrist_image))
    else:
        print("Using dummy images (provide --context_image and --wrist_image for real data)")
        context_img, wrist_img = create_dummy_images()

    # Run visualization
    if args.mode == "actions":
        visualize_actions(
            model, context_img, wrist_img, args.instruction,
            args.num_samples, output_dir, args.device
        )
    elif args.mode == "flow":
        visualize_flow(
            model, context_img, wrist_img, args.instruction,
            args.num_flow_steps, output_dir, args.device
        )
    elif args.mode == "attention":
        visualize_attention(
            model, context_img, wrist_img, args.instruction,
            output_dir, args.device
        )

    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
