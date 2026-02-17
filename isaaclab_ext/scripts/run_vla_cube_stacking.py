#!/usr/bin/env python3
"""
Run Trained VLA Model in Isaac Lab 3-Cube Stacking Environment

Loads a trained VLA checkpoint and runs closed-loop inference in the Isaac Lab
cube stacking scene (Franka + 3 cubes + table_cam/wrist_cam).  By default
launches Isaac Sim with the GUI so you can watch the robot attempt the task.

Usage:
    # With GUI (default)
    python isaaclab_ext/scripts/run_vla_cube_stacking.py \
        --checkpoint checkpoints/cube_stacking_cosmos/step_29000.pt \
        --hdf5_path data/cosmos_dataset_1k.hdf5 \
        --key_mapping_file configs/key_mappings/nvidia_cosmos.json

    # Headless evaluation
    python isaaclab_ext/scripts/run_vla_cube_stacking.py \
        --checkpoint checkpoints/cube_stacking_cosmos/step_29000.pt \
        --hdf5_path data/cosmos_dataset_1k.hdf5 \
        --key_mapping_file configs/key_mappings/nvidia_cosmos.json \
        --headless --num_episodes 20
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing MUST happen before IsaacSim / AppLauncher imports
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(
    description="Run trained VLA in Isaac Lab cube stacking env"
)
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to trained VLA checkpoint (.pt)",
)
parser.add_argument(
    "--hdf5_path", type=str, default=None,
    help="Path to HDF5 dataset (for proprio normalization stats fallback)",
)
parser.add_argument(
    "--key_mapping_file", type=str,
    default="configs/key_mappings/nvidia_cosmos.json",
    help="Path to JSON key mapping file",
)
parser.add_argument(
    "--instruction", type=str,
    default="Stack the cubes in order: place the red cube on the blue cube, "
            "then place the green cube on the red cube.",
    help="Task instruction for the VLA model",
)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--vlm_model", type=str,
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--action_dim", type=int, default=7)
parser.add_argument("--chunk_size", type=int, default=16)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--proprio_dim", type=int, default=27)

# Append AppLauncher cli args and parse
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Launch Isaac Sim (must happen before any other Isaac Lab imports)
# ---------------------------------------------------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All remaining imports (Isaac Lab + VLA) go here
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

# Isaac Lab
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_env_cfg import (  # noqa: E402
    FrankaCubeStackVisuomotorEnvCfg,
)

# Add VLA project root to path
VLA_WORK = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(VLA_WORK))

from vla.models import create_vla_model  # noqa: E402
from vla.utils.transforms import ActionChunker  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: load normalization stats (proprio + action)
# ---------------------------------------------------------------------------
def load_normalization_stats(checkpoint_path: str, hdf5_path: str | None,
                             key_mapping: dict | None):
    """Load proprio and action mean/std — checkpoint first, else from HDF5."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    proprio_mean = ckpt.get("proprio_mean")
    proprio_std = ckpt.get("proprio_std")

    # Action stats: the model's action_mean/std may be zeros/ones if the
    # dataset handled normalization but the stats were never bridged to the
    # model.  Check for dataset-level stats saved in the checkpoint first.
    action_mean = ckpt.get("dataset_action_mean")
    action_std = ckpt.get("dataset_action_std")

    all_found = (proprio_mean is not None and proprio_std is not None
                 and action_mean is not None and action_std is not None)

    if all_found:
        print("Loaded all normalization stats from checkpoint")
        return (np.asarray(proprio_mean, dtype=np.float32),
                np.asarray(proprio_std, dtype=np.float32),
                np.asarray(action_mean, dtype=np.float32),
                np.asarray(action_std, dtype=np.float32))

    # Fallback: compute from HDF5 dataset
    if hdf5_path is None:
        print("WARNING: Incomplete normalization stats and no --hdf5_path.")
        return (None, None, None, None)

    print(f"Computing normalization stats from {hdf5_path} ...")
    from vla.data import HDF5VLADataset  # noqa: E402
    dataset = HDF5VLADataset(
        hdf5_path,
        instruction="",
        chunk_size=16,
        augment=False,
        key_mapping=key_mapping,
    )
    stats = dataset.get_stats()
    return (stats["proprio_mean"], stats["proprio_std"],
            stats["action_mean"], stats["action_std"])


# ---------------------------------------------------------------------------
# Helper: extract observations from env obs dict
# ---------------------------------------------------------------------------
def extract_obs(obs: dict, proprio_mean, proprio_std, device):
    """
    Extract camera images and proprio state from the Isaac Lab observation
    dictionary returned by env.reset() / env.step().

    The visuomotor env config uses ``concatenate_terms = False`` so
    ``obs["policy"]`` is a dict of individual observation tensors.

    Returns:
        context_pil: PIL Image (224x224)
        wrist_pil:   PIL Image (224x224)
        proprio_tensor: [1, 27] bfloat16 tensor (or None)
    """
    policy = obs["policy"]

    # --- Images ----------------------------------------------------------
    # Observation images are [num_envs, H, W, C] uint8 tensors on GPU
    table_cam = policy["table_cam"][0].cpu().numpy()   # [H, W, C] uint8
    wrist_cam = policy["wrist_cam"][0].cpu().numpy()    # [H, W, C] uint8

    # Resize from env resolution (200x200) to VLA input (224x224)
    context_pil = PILImage.fromarray(table_cam).resize(
        (224, 224), PILImage.BILINEAR
    )
    wrist_pil = PILImage.fromarray(wrist_cam).resize(
        (224, 224), PILImage.BILINEAR
    )

    # --- Proprioception ---------------------------------------------------
    # Each key is [num_envs, dim]; take env 0
    joint_pos = policy["joint_pos"][0].cpu().numpy()     # [9]
    joint_vel = policy["joint_vel"][0].cpu().numpy()     # [9]
    eef_pos   = policy["eef_pos"][0].cpu().numpy()       # [3]
    eef_quat  = policy["eef_quat"][0].cpu().numpy()      # [4]
    gripper   = policy["gripper_pos"][0].cpu().numpy()   # [2]
    proprio = np.concatenate([joint_pos, joint_vel, eef_pos, eef_quat, gripper])

    # Normalize
    if proprio_mean is not None and proprio_std is not None:
        proprio = (proprio - proprio_mean) / (proprio_std + 1e-8)

    proprio_tensor = (
        torch.tensor(proprio, dtype=torch.bfloat16)
        .unsqueeze(0)
        .to(device)
    )

    return context_pil, wrist_pil, proprio_tensor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device(args.device)

    # --- Load key mapping ------------------------------------------------
    key_mapping = None
    km_path = VLA_WORK / args.key_mapping_file
    if km_path.exists():
        with open(km_path, "r") as f:
            key_mapping = json.load(f)
        print(f"Key mapping: {km_path}")

    # --- Load normalization stats (proprio + action) ---------------------
    ckpt_path = str(VLA_WORK / args.checkpoint)
    hdf5_path = str(VLA_WORK / args.hdf5_path) if args.hdf5_path else None
    proprio_mean, proprio_std, action_mean, action_std = \
        load_normalization_stats(ckpt_path, hdf5_path, key_mapping)

    if action_mean is not None:
        print(f"Action mean: {action_mean}")
        print(f"Action std:  {action_std}")

    # --- Create Isaac Lab environment ------------------------------------
    print("\nCreating cube stacking environment ...")
    env_cfg = FrankaCubeStackVisuomotorEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"Environment created.  Action space dim: "
          f"{env.action_manager.total_action_dim}")

    # --- Load VLA model --------------------------------------------------
    print("\nLoading VLA model ...")
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
    model.load_checkpoint(ckpt_path)

    # Cast trainable components to bfloat16 to match VLM backbone output
    model.projector.to(device=device, dtype=torch.bfloat16)
    model.action_expert.to(device=device, dtype=torch.bfloat16)
    model.flow_head.to(device=device, dtype=torch.bfloat16)
    # Move top-level registered buffers (action_mean, action_std) to GPU
    model.action_mean = model.action_mean.to(device)
    model.action_std = model.action_std.to(device)
    model.eval()
    print("VLA model loaded and ready.", flush=True)

    # --- Action chunker for temporal smoothing ---------------------------
    chunker = ActionChunker(chunk_size=args.chunk_size, overlap=4)

    # --- Run episodes ----------------------------------------------------
    episode_results = []

    for ep in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.num_episodes}")
        print(f"{'='*60}")

        obs, info = env.reset()
        chunker.reset()

        episode_reward = 0.0
        subtask_reached = {"grasp_1": False, "stack_1": False, "grasp_2": False}
        needs_new_chunk = True  # request prediction on first step

        for step in range(args.max_steps):
            # Extract observations
            context_pil, wrist_pil, proprio_tensor = extract_obs(
                obs, proprio_mean, proprio_std, device
            )

            # Predict new action chunk when needed
            if needs_new_chunk:
                with torch.no_grad():
                    action_chunk = model.predict_action(
                        images=[context_pil, wrist_pil],
                        instruction=args.instruction,
                        proprio_state=proprio_tensor,
                    )
                if step == 0:
                    print(f"  First prediction: shape={action_chunk.shape}", flush=True)
                # action_chunk: [1, chunk_size, action_dim] — in normalized space
                chunk_np = action_chunk[0].float().cpu().numpy()

                # Denormalize from dataset-normalized space to raw action space.
                # The model's action_mean/std are 0/1 (dataset did normalization),
                # so we must manually map back to the env's expected scale.
                if action_mean is not None and action_std is not None:
                    chunk_np = chunk_np * action_std + action_mean

                needs_new_chunk = False

            # Get single action from chunker
            action = chunker.update(chunk_np)

            # Check if chunker needs refilling next step
            remaining = chunker.get_current_chunk()
            if remaining is None or len(remaining) <= 1:
                needs_new_chunk = True

            # Clip and send to env
            action_clipped = np.clip(action, -1.0, 1.0)
            action_tensor = (
                torch.tensor(action_clipped, dtype=torch.float32)
                .unsqueeze(0)
                .to(env.device)
            )

            obs, reward, terminated, truncated, info = env.step(action_tensor)
            episode_reward += reward.item() if torch.is_tensor(reward) else reward

            # Track subtask progress
            if "subtask_terms" in obs:
                st = obs["subtask_terms"]
                for key in subtask_reached:
                    if key in st and st[key][0].item() > 0.5:
                        if not subtask_reached[key]:
                            subtask_reached[key] = True
                            print(f"  Step {step:3d}: subtask '{key}' completed!", flush=True)

            # Print progress periodically
            if (step + 1) % 50 == 0:
                act_str = " ".join(f"{a:.3f}" for a in action_clipped[:3])
                print(f"  Step {step+1:3d}: action_xyz=[{act_str}]  "
                      f"reward={episode_reward:.2f}", flush=True)

            # Check termination
            if terminated.any() if torch.is_tensor(terminated) else terminated:
                print(f"  Episode terminated at step {step+1}", flush=True)
                break
            if truncated.any() if torch.is_tensor(truncated) else truncated:
                print(f"  Episode truncated at step {step+1}", flush=True)
                break

        # Episode summary
        success = subtask_reached.get("stack_1", False)
        full_success = all(subtask_reached.values())
        print(f"\n  Reward: {episode_reward:.2f}", flush=True)
        print(f"  Subtasks: {subtask_reached}", flush=True)
        print(f"  Stack-1 success: {success}", flush=True)
        print(f"  Full success (all subtasks): {full_success}", flush=True)

        episode_results.append({
            "episode": ep,
            "reward": episode_reward,
            "subtasks": dict(subtask_reached),
            "success": success,
            "full_success": full_success,
        })

    # --- Summary ---------------------------------------------------------
    print(f"\n{'='*60}", flush=True)
    print("Evaluation Summary", flush=True)
    print(f"{'='*60}", flush=True)
    n = len(episode_results)
    success_count = sum(1 for r in episode_results if r["success"])
    full_count = sum(1 for r in episode_results if r["full_success"])
    avg_reward = np.mean([r["reward"] for r in episode_results])
    print(f"Episodes:       {n}", flush=True)
    print(f"Stack-1 rate:   {success_count}/{n} ({100*success_count/n:.1f}%)", flush=True)
    print(f"Full success:   {full_count}/{n} ({100*full_count/n:.1f}%)", flush=True)
    print(f"Avg reward:     {avg_reward:.2f}", flush=True)
    print(f"{'='*60}", flush=True)
    sys.stdout.flush()

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
