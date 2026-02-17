#!/usr/bin/env python3
"""
VLA Overfit Test

Trains the VLA on 5 demos until near-zero loss, then evaluates open-loop
on those same demos to confirm the full pipeline works end-to-end.

If the model can't overfit 5 demos → pipeline/data wiring bug.
If it overfits but fails closed-loop → compounding error / distribution shift.

Usage:
    python scripts/overfit_test.py \
        --hdf5_path data/cosmos_dataset_1k.hdf5 \
        --key_mapping_file configs/key_mappings/nvidia_cosmos.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla.models import create_vla_model
from vla.data import HDF5VLADataset


def parse_args():
    parser = argparse.ArgumentParser(description="VLA overfit test")
    parser.add_argument("--hdf5_path", type=str, required=True)
    parser.add_argument("--key_mapping_file", type=str,
                        default="configs/key_mappings/nvidia_cosmos.json")
    parser.add_argument("--instruction", type=str,
                        default="Stack the cubes in order: place the red cube on the blue cube, "
                                "then place the green cube on the red cube.")
    parser.add_argument("--num_demos", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=50000,
                        help="Max training steps")
    parser.add_argument("--target_loss", type=float, default=0.10,
                        help="Stop training when loss reaches this")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--vlm_model", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints/overfit_test")
    return parser.parse_args()


def tensor_to_pil(img_tensor):
    """Convert [C, H, W] float32 tensor in [0,1] to PIL Image."""
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return PILImage.fromarray(img)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- Load key mapping ---
    key_mapping = None
    km_path = PROJECT_ROOT / args.key_mapping_file
    if km_path.exists():
        with open(km_path) as f:
            key_mapping = json.load(f)

    # =================================================================
    # Phase 1: Create tiny dataset (5 demos)
    # =================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: Loading {args.num_demos}-demo dataset")
    print(f"{'='*70}")

    hdf5_path = str(PROJECT_ROOT / args.hdf5_path)
    ds_kwargs = dict(
        instruction=args.instruction,
        chunk_size=16,
        augment=False,  # No augmentation for overfit test
        max_demos=args.num_demos,
    )
    if key_mapping:
        ds_kwargs["key_mapping"] = key_mapping

    dataset = HDF5VLADataset(hdf5_path, **ds_kwargs)
    action_mean = dataset.action_mean
    action_std = dataset.action_std

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Simple, no multiprocessing for small dataset
        drop_last=False,
    )

    print(f"Demos: {args.num_demos}")
    print(f"Samples: {len(dataset)}")
    print(f"Batches/epoch: {len(loader)}")
    print(f"Action mean: {action_mean}")
    print(f"Action std:  {action_std}")

    # =================================================================
    # Phase 2: Create model (fresh, no checkpoint)
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Creating fresh VLA model")
    print(f"{'='*70}")

    model = create_vla_model(
        model_name=args.vlm_model,
        action_dim=7, chunk_size=16, hidden_dim=512, proprio_dim=27,
        freeze_vlm=True, use_lora=True, device_map="auto",
    )

    # Store dataset stats for checkpointing
    model._proprio_mean = dataset.proprio_mean
    model._proprio_std = dataset.proprio_std
    model._dataset_action_mean = dataset.action_mean
    model._dataset_action_std = dataset.action_std

    # Move trainable components to GPU (VLM is already placed by device_map="auto")
    model.projector.to(device)
    model.action_expert.to(device)
    model.flow_head.to(device)
    model.action_mean = model.action_mean.to(device)
    model.action_std = model.action_std.to(device)

    trainable_params = model.get_trainable_parameters()
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_trainable:,}")

    # =================================================================
    # Phase 3: Training loop (overfit)
    # =================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 3: Training to overfit (target loss: {args.target_loss})")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    # Simple linear warmup + constant schedule
    warmup_steps = 100

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    loss_history = []
    start_time = time.time()

    for epoch in range(9999):  # Unlimited epochs, stop by loss or max_steps
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch in loader:
            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # LR warmup
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            # Forward
            with autocast("cuda", enabled=True, dtype=torch.bfloat16):
                instructions = batch["instruction"]
                if isinstance(instructions, str):
                    instructions = [instructions] * batch["context_image"].shape[0]

                outputs = model(
                    context_images=batch["context_image"],
                    wrist_images=batch["wrist_image"],
                    instructions=instructions,
                    proprio_state=batch.get("proprio_state"),
                    target_actions=batch["actions"],
                )
                loss = outputs["loss"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_batches += 1
            global_step += 1
            loss_history.append(loss_val)

            # Log every 50 steps
            if global_step % 50 == 0:
                avg_recent = np.mean(loss_history[-50:])
                elapsed = time.time() - start_time
                print(f"  Step {global_step:5d} | Loss: {loss_val:.4f} | "
                      f"Avg(50): {avg_recent:.4f} | "
                      f"Time: {elapsed:.0f}s")

            # Check convergence
            if len(loss_history) >= 20:
                recent_avg = np.mean(loss_history[-20:])
                if recent_avg <= args.target_loss:
                    print(f"\n  Target loss {args.target_loss} reached at step {global_step}!")
                    break

            if global_step >= args.max_steps:
                print(f"\n  Max steps ({args.max_steps}) reached.")
                break

        # End-of-epoch summary
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        print(f"Epoch {epoch+1} | Avg loss: {avg_epoch_loss:.4f} | Steps: {global_step}")

        # Save best checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ckpt_path = ckpt_dir / "overfit_best.pt"
            checkpoint = {
                "checkpoint_type": "lightweight",
                "projector_state_dict": model.projector.state_dict(),
                "action_expert_state_dict": model.action_expert.state_dict(),
                "flow_head_state_dict": model.flow_head.state_dict(),
                "action_mean": model.action_mean,
                "action_std": model.action_std,
                "dataset_action_mean": model._dataset_action_mean,
                "dataset_action_std": model._dataset_action_std,
                "proprio_mean": model._proprio_mean,
                "proprio_std": model._proprio_std,
                "global_step": global_step,
                "epoch": epoch,
            }
            if hasattr(model.vision_encoder, "lora_modules"):
                checkpoint["lora_state_dict"] = \
                    model.vision_encoder.lora_modules.state_dict()
            torch.save(checkpoint, ckpt_path)

        # Check stop conditions
        if len(loss_history) >= 20 and np.mean(loss_history[-20:]) <= args.target_loss:
            break
        if global_step >= args.max_steps:
            break

    elapsed = time.time() - start_time
    print(f"\nTraining complete: {global_step} steps in {elapsed:.0f}s")
    print(f"Final avg loss (last 20): {np.mean(loss_history[-20:]):.4f}")
    print(f"Best epoch loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {ckpt_dir / 'overfit_best.pt'}")

    # =================================================================
    # Phase 4: Open-loop evaluation on the SAME 5 demos
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 4: Open-loop evaluation on training demos")
    print(f"{'='*70}")

    model.eval()

    # Move trainable components for inference
    model.projector.to(device=device, dtype=torch.bfloat16)
    model.action_expert.to(device=device, dtype=torch.bfloat16)
    model.flow_head.to(device=device, dtype=torch.bfloat16)
    model.action_mean = model.action_mean.to(device)
    model.action_std = model.action_std.to(device)

    # Evaluate on every 10th sample from the dataset (to keep it fast)
    eval_indices = list(range(0, len(dataset), max(1, len(dataset) // 50)))[:50]

    all_errors_first = []
    all_errors_chunk = []

    for i, idx in enumerate(eval_indices):
        sample = dataset[idx]

        # Ground truth (normalized by dataset, denormalize for comparison)
        gt_norm = sample["actions"].numpy()  # [16, 7]
        gt_raw = gt_norm * action_std + action_mean

        # Prepare inputs
        context_pil = tensor_to_pil(sample["context_image"])
        wrist_pil = tensor_to_pil(sample["wrist_image"])
        proprio_t = sample["proprio_state"].unsqueeze(0).to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            pred_chunk = model.predict_action(
                images=[context_pil, wrist_pil],
                instruction=args.instruction,
                proprio_state=proprio_t,
            )
        pred_norm = pred_chunk[0].float().cpu().numpy()
        pred_raw = pred_norm * action_std + action_mean

        # Errors
        first_err = np.abs(pred_raw[0] - gt_raw[0])
        chunk_err = np.abs(pred_raw - gt_raw).mean(axis=0)
        all_errors_first.append(first_err)
        all_errors_chunk.append(chunk_err)

        if i < 5:
            print(f"\nSample {i+1} (idx={idx}):")
            print(f"  GT  [0]: {gt_raw[0]}")
            print(f"  Pred[0]: {pred_raw[0]}")
            print(f"  Err [0]: {first_err}")

    all_errors_first = np.array(all_errors_first)
    all_errors_chunk = np.array(all_errors_chunk)

    print(f"\n--- Open-loop Evaluation Results ({len(eval_indices)} samples) ---")
    print(f"First-step MAE per dim: {all_errors_first.mean(axis=0)}")
    print(f"First-step MAE overall: {all_errors_first.mean():.6f}")
    print(f"Chunk MAE per dim:      {all_errors_chunk.mean(axis=0)}")
    print(f"Chunk MAE overall:      {all_errors_chunk.mean():.6f}")

    # Compare to action scale
    print(f"\nAction std (reference): {action_std}")
    relative_err = all_errors_first.mean(axis=0) / (action_std + 1e-10)
    print(f"Relative error (MAE/std): {relative_err}")
    print(f"Mean relative error:      {relative_err.mean():.4f}")

    # =================================================================
    # Verdict
    # =================================================================
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    final_loss = np.mean(loss_history[-20:]) if len(loss_history) >= 20 else best_loss
    mean_rel_err = relative_err[:6].mean()  # Position/rotation dims

    if final_loss < 0.1 and mean_rel_err < 0.3:
        print("PASS: Model can overfit 5 demos. Pipeline is correct.")
        print("The full-dataset model's poor performance is due to:")
        print("  - Insufficient training (loss ~1.0, needs ~0.3-0.5)")
        print("  - Compounding errors in closed-loop rollout")
        print("  - Possibly not enough data diversity")
        print("\nNext steps:")
        print("  1. Train on full dataset for longer (aim for loss ~0.3-0.5)")
        print("  2. Try replanning every 1-4 steps instead of executing full 16-step chunks")
    elif final_loss < 0.3:
        print("PARTIAL: Model partly overfits but predictions aren't precise.")
        print("This suggests the architecture can learn but may need:")
        print("  - More training steps for overfit test")
        print("  - Higher learning rate")
        print("  - Check that VLM features vary across samples (not constant)")
    else:
        print("FAIL: Model cannot overfit 5 demos.")
        print("This indicates a pipeline/wiring bug:")
        print("  - Check action normalization consistency")
        print("  - Check image processing (train vs inference)")
        print("  - Check that gradients flow through all components")
        print("  - Try training with augment=False and higher lr")


if __name__ == "__main__":
    main()
