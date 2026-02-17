#!/usr/bin/env python3
"""
Cube Stacking Inference Script

Loads a trained VLA checkpoint and runs inference on sample images from the
NVIDIA cosmos dataset. Prints predicted 7D action chunks for verification.

Usage:
    python scripts/run_cube_stacking.py \
        --checkpoint checkpoints/cube_stacking_cosmos/best.pt \
        --hdf5_path data/cosmos_dataset_1k.hdf5

    python scripts/run_cube_stacking.py \
        --checkpoint checkpoints/cube_stacking_cosmos/best.pt \
        --hdf5_path data/cosmos_dataset_1k.hdf5 \
        --num_samples 5 --visualize
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vla.models import VLAModel, create_vla_model


def load_key_mapping(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run cube stacking inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--hdf5_path", type=str,
                        default="data/cosmos_dataset_1k.hdf5",
                        help="Path to HDF5 dataset for sample images")
    parser.add_argument("--key_mapping_file", type=str,
                        default="configs/key_mappings/nvidia_cosmos.json",
                        help="Path to key mapping JSON")
    parser.add_argument("--instruction", type=str,
                        default="Stack the cubes in order: place the red cube on the blue cube, then place the green cube on the red cube.",
                        help="Task instruction")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of demo timesteps to run inference on")
    parser.add_argument("--vlm_model", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="VLM model name (must match training)")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--proprio_dim", type=int, default=27)
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization of input images and predicted actions")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load key mapping
    key_mapping = load_key_mapping(args.key_mapping_file)
    context_key = key_mapping.get("context_rgb", "agentview_rgb")
    wrist_key = key_mapping.get("wrist_rgb", "eye_in_hand_rgb")

    # Build proprio keys from mapping
    canonical_proprio = ["joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_pos"]
    proprio_keys = [key_mapping[k] for k in canonical_proprio if k in key_mapping]

    print("=" * 60)
    print("VLA Cube Stacking Inference")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.hdf5_path}")
    print(f"Instruction: {args.instruction}")
    print(f"Context camera key: {context_key}")
    print(f"Wrist camera key: {wrist_key}")
    print(f"Proprio keys: {proprio_keys}")

    # Create model
    print("\nLoading VLA model...")
    model = create_vla_model(
        model_name=args.vlm_model,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        proprio_dim=args.proprio_dim,
        freeze_vlm=True,
        use_lora=True,
        device_map="auto",
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_checkpoint(args.checkpoint)
    device = torch.device(args.device)

    # Move trainable components to GPU with matching dtype (VLM outputs bfloat16)
    model.projector.to(device=device, dtype=torch.bfloat16)
    model.action_expert.to(device=device, dtype=torch.bfloat16)
    model.flow_head.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Load sample images from dataset
    print(f"\nLoading samples from {args.hdf5_path}...")
    with h5py.File(args.hdf5_path, 'r') as f:
        data_group = f['data']
        demo_names = sorted([k for k in data_group.keys() if k.startswith('demo_')])

        if not demo_names:
            print("ERROR: No demos found in dataset")
            return 1

        results = []
        for sample_idx in range(min(args.num_samples, len(demo_names))):
            demo_name = demo_names[sample_idx]
            demo = data_group[demo_name]
            obs = demo['obs']

            # Pick a timestep in the middle of the episode
            ep_length = demo[key_mapping.get("actions", "actions")].shape[0]
            t = ep_length // 3  # early-ish in the episode

            # Load images
            context_img = obs[context_key][t]  # [H, W, 3] uint8
            wrist_img = obs[wrist_key][t]      # [H, W, 3] uint8

            # Load proprioception
            proprio_parts = []
            for pk in proprio_keys:
                if pk in obs:
                    proprio_parts.append(obs[pk][t])
            proprio_state = np.concatenate(proprio_parts, axis=-1).astype(np.float32) if proprio_parts else None

            # Load ground-truth actions for comparison
            action_key = key_mapping.get("actions", "actions")
            end_t = min(t + args.chunk_size, ep_length)
            gt_actions = demo[action_key][t:end_t]

            # Process images to float tensors
            from PIL import Image as PILImage
            target_size = (224, 224)

            def process_img(img):
                if img.shape[:2] != target_size:
                    img = np.array(PILImage.fromarray(img).resize(
                        (target_size[1], target_size[0]), PILImage.BILINEAR
                    ))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # [C, H, W]
                return torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, C, H, W]

            context_tensor = process_img(context_img).to(device)
            wrist_tensor = process_img(wrist_img).to(device)

            proprio_tensor = None
            if proprio_state is not None:
                proprio_tensor = torch.tensor(proprio_state, dtype=torch.bfloat16).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(
                    context_images=context_tensor,
                    wrist_images=wrist_tensor,
                    instructions=[args.instruction],
                    proprio_state=proprio_tensor,
                )

            predicted_actions = outputs['actions'].cpu().numpy()[0]  # [chunk_size, action_dim]

            print(f"\n{'─' * 60}")
            print(f"Sample {sample_idx}: {demo_name}, timestep {t}/{ep_length}")
            print(f"{'─' * 60}")
            print(f"Predicted actions shape: {predicted_actions.shape}")
            print(f"Predicted first 3 steps:")
            for step in range(min(3, predicted_actions.shape[0])):
                a = predicted_actions[step]
                print(f"  t+{step}: pos=[{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}] "
                      f"rot=[{a[3]:.4f}, {a[4]:.4f}, {a[5]:.4f}] "
                      f"grip={a[6]:.4f}")

            print(f"Ground truth first 3 steps:")
            for step in range(min(3, gt_actions.shape[0])):
                a = gt_actions[step]
                print(f"  t+{step}: pos=[{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}] "
                      f"rot=[{a[3]:.4f}, {a[4]:.4f}, {a[5]:.4f}] "
                      f"grip={a[6]:.4f}")

            # Compute L2 error
            min_len = min(predicted_actions.shape[0], gt_actions.shape[0])
            l2_error = np.sqrt(np.mean((predicted_actions[:min_len] - gt_actions[:min_len]) ** 2))
            print(f"RMSE (pred vs GT): {l2_error:.4f}")

            results.append({
                'demo': demo_name,
                'timestep': t,
                'rmse': float(l2_error),
                'predicted': predicted_actions,
                'ground_truth': gt_actions,
                'context_img': context_img,
                'wrist_img': wrist_img,
            })

    # Summary
    avg_rmse = np.mean([r['rmse'] for r in results])
    print(f"\n{'=' * 60}")
    print(f"Average RMSE across {len(results)} samples: {avg_rmse:.4f}")
    print(f"{'=' * 60}")

    # Visualize if requested
    if args.visualize and results:
        try:
            import matplotlib.pyplot as plt

            n = len(results)
            fig, axes = plt.subplots(3, n, figsize=(5 * n, 12))
            if n == 1:
                axes = axes.reshape(-1, 1)

            for i, r in enumerate(results):
                # Context image
                axes[0, i].imshow(r['context_img'])
                axes[0, i].set_title(f"{r['demo']} t={r['timestep']}\nContext")
                axes[0, i].axis('off')

                # Wrist image
                axes[1, i].imshow(r['wrist_img'])
                axes[1, i].set_title("Wrist")
                axes[1, i].axis('off')

                # Action trajectory comparison (first 3 dims: xyz position)
                pred = r['predicted'][:, :3]
                gt = r['ground_truth'][:, :3]
                for dim, label in enumerate(['x', 'y', 'z']):
                    axes[2, i].plot(pred[:, dim], '--', label=f'pred_{label}')
                    axes[2, i].plot(gt[:min(len(gt), len(pred)), dim], '-', label=f'gt_{label}')
                axes[2, i].set_title(f"Actions (RMSE={r['rmse']:.3f})")
                axes[2, i].legend(fontsize=6)
                axes[2, i].set_xlabel("Step")

            plt.tight_layout()
            out_path = "checkpoints/cube_stacking_cosmos/inference_results.png"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150)
            print(f"\nVisualization saved to {out_path}")
            plt.close()
        except ImportError:
            print("matplotlib not available, skipping visualization")

    return 0


if __name__ == "__main__":
    sys.exit(main())
