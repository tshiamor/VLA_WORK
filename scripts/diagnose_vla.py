#!/usr/bin/env python3
"""
VLA Diagnostic Script

Runs key checks to identify why the model underperforms:
1. Action stats comparison (predicted vs dataset)
2. Vision ablation (real images vs zeros → is vision used?)
3. Per-component feature norms (VLM → projector → action_expert → flow_head)
4. Open-loop prediction vs ground truth on dataset samples
5. Mode collapse check (sample N actions for same observation)

Usage:
    python scripts/diagnose_vla.py \
        --checkpoint checkpoints/cube_stacking_cosmos/step_29000.pt \
        --hdf5_path data/cosmos_dataset_1k.hdf5 \
        --key_mapping_file configs/key_mappings/nvidia_cosmos.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla.models import create_vla_model
from vla.data import HDF5VLADataset


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose VLA model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hdf5_path", type=str, required=True)
    parser.add_argument("--key_mapping_file", type=str,
                        default="configs/key_mappings/nvidia_cosmos.json")
    parser.add_argument("--instruction", type=str,
                        default="Stack the cubes in order: place the red cube on the blue cube, "
                                "then place the green cube on the red cube.")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of dataset samples to test")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def load_model(args, device):
    """Load model with proper dtype/device handling."""
    model = create_vla_model(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        action_dim=7, chunk_size=16, hidden_dim=512, proprio_dim=27,
        freeze_vlm=True, use_lora=True, device_map="auto",
    )
    model.load_checkpoint(str(PROJECT_ROOT / args.checkpoint))
    model.projector.to(device=device, dtype=torch.bfloat16)
    model.action_expert.to(device=device, dtype=torch.bfloat16)
    model.flow_head.to(device=device, dtype=torch.bfloat16)
    model.action_mean = model.action_mean.to(device)
    model.action_std = model.action_std.to(device)
    model.eval()
    return model


def tensor_to_pil(img_tensor):
    """Convert [C, H, W] float32 tensor to PIL Image."""
    img = img_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return PILImage.fromarray(img)


def get_feature_norms(model, context_pil, wrist_pil, instruction, proprio_tensor):
    """Get intermediate feature norms through the pipeline."""
    with torch.no_grad():
        # Step 1: VLM
        vlm_outputs = model.vision_encoder(
            images=[context_pil, wrist_pil],
            instruction=instruction,
            return_last_hidden_state=True,
        )
        vlm_emb = vlm_outputs.get('last_hidden_state',
                                   vlm_outputs.get('pooled_output'))
        vlm_norm = vlm_emb.float().norm().item()

        # Step 2: Projector
        projected = model.projector(vlm_emb)
        proj_norm = projected.float().norm().item()

        # Step 3: Action expert
        expert_out = model.action_expert(
            vlm_embeddings=projected,
            proprio_state=proprio_tensor,
        )
        condition = expert_out['condition']
        cond_norm = condition.float().norm().item()
        feat_norm = expert_out['action_features'].float().norm().item()

        # Step 4: Flow head sample
        flow_dtype = next(model.flow_head.parameters()).dtype
        condition_cast = condition.to(dtype=flow_dtype)
        actions = model.flow_head.sample(condition_cast)
        action_norm = actions.float().norm().item()

    return {
        'vlm_emb': vlm_norm,
        'projected': proj_norm,
        'condition': cond_norm,
        'action_features': feat_norm,
        'sampled_actions': action_norm,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- Load key mapping ---
    key_mapping = None
    km_path = PROJECT_ROOT / args.key_mapping_file
    if km_path.exists():
        with open(km_path) as f:
            key_mapping = json.load(f)

    # --- Load dataset ---
    print("Loading dataset...")
    dataset = HDF5VLADataset(
        str(PROJECT_ROOT / args.hdf5_path),
        instruction=args.instruction,
        chunk_size=16, augment=False, key_mapping=key_mapping,
    )
    action_mean = dataset.action_mean
    action_std = dataset.action_std
    proprio_mean = dataset.proprio_mean
    proprio_std = dataset.proprio_std

    print(f"\nDataset action_mean: {action_mean}")
    print(f"Dataset action_std:  {action_std}")

    # --- Load model ---
    print("\nLoading model...")
    model = load_model(args, device)

    # Check model's internal action stats
    print(f"\nModel action_mean: {model.action_mean.cpu().numpy()}")
    print(f"Model action_std:  {model.action_std.cpu().numpy()}")

    # =====================================================================
    # TEST 1: Open-loop prediction vs ground truth
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Open-loop prediction vs ground truth")
    print(f"{'='*70}")

    all_pred_raw = []
    all_gt_raw = []
    all_errors = []

    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)),
                               replace=False)

    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]

        # Ground truth (already normalized by dataset)
        gt_normalized = sample['actions'].numpy()  # [16, 7]
        gt_raw = gt_normalized * action_std + action_mean

        # Prepare inputs for model
        context_pil = tensor_to_pil(sample['context_image'])
        wrist_pil = tensor_to_pil(sample['wrist_image'])
        proprio_tensor = sample['proprio_state'].unsqueeze(0).to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            pred_chunk = model.predict_action(
                images=[context_pil, wrist_pil],
                instruction=args.instruction,
                proprio_state=proprio_tensor,
            )
        # pred_chunk is [1, 16, 7] in normalized space (model mean=0, std=1)
        pred_normalized = pred_chunk[0].float().cpu().numpy()
        pred_raw = pred_normalized * action_std + action_mean

        all_pred_raw.append(pred_raw)
        all_gt_raw.append(gt_raw)

        # Per-step error (first action in chunk is most important)
        first_step_error = np.abs(pred_raw[0] - gt_raw[0])
        all_errors.append(first_step_error)

        if i < 3:  # Show first 3 samples
            print(f"\nSample {i+1} (dataset idx {idx}):")
            print(f"  GT  action[0]: {gt_raw[0]}")
            print(f"  Pred action[0]: {pred_raw[0]}")
            print(f"  Abs error:      {first_step_error}")

    all_pred_raw = np.array(all_pred_raw)  # [N, 16, 7]
    all_gt_raw = np.array(all_gt_raw)      # [N, 16, 7]
    all_errors = np.array(all_errors)       # [N, 7]

    print(f"\nFirst-step MAE per dim: {all_errors.mean(axis=0)}")
    print(f"First-step MAE overall: {all_errors.mean():.6f}")

    # Compare distributions
    print(f"\nPredicted action stats (raw, first step):")
    print(f"  mean: {all_pred_raw[:, 0, :].mean(axis=0)}")
    print(f"  std:  {all_pred_raw[:, 0, :].std(axis=0)}")
    print(f"Ground truth action stats (raw, first step):")
    print(f"  mean: {all_gt_raw[:, 0, :].mean(axis=0)}")
    print(f"  std:  {all_gt_raw[:, 0, :].std(axis=0)}")

    # Check if predictions are near-constant (mode collapse)
    pred_variance = all_pred_raw[:, 0, :].var(axis=0)
    gt_variance = all_gt_raw[:, 0, :].var(axis=0)
    variance_ratio = pred_variance / (gt_variance + 1e-10)
    print(f"\nVariance ratio (pred/gt) per dim: {variance_ratio}")
    print(f"  (<<1 = mode collapse, ~1 = good, >>1 = too noisy)")

    # =====================================================================
    # TEST 2: Vision ablation (real images vs zeros)
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 2: Vision ablation (real images vs zeros/noise)")
    print(f"{'='*70}")

    sample = dataset[0]
    context_pil = tensor_to_pil(sample['context_image'])
    wrist_pil = tensor_to_pil(sample['wrist_image'])
    proprio_tensor = sample['proprio_state'].unsqueeze(0).to(device, dtype=torch.bfloat16)

    # Real images
    with torch.no_grad():
        pred_real = model.predict_action(
            images=[context_pil, wrist_pil],
            instruction=args.instruction,
            proprio_state=proprio_tensor,
        )[0].float().cpu().numpy()

    # Black images
    black_img = PILImage.new('RGB', (224, 224), (0, 0, 0))
    with torch.no_grad():
        pred_black = model.predict_action(
            images=[black_img, black_img],
            instruction=args.instruction,
            proprio_state=proprio_tensor,
        )[0].float().cpu().numpy()

    # Random noise images
    noise_img = PILImage.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    with torch.no_grad():
        pred_noise = model.predict_action(
            images=[noise_img, noise_img],
            instruction=args.instruction,
            proprio_state=proprio_tensor,
        )[0].float().cpu().numpy()

    # Compare (first step)
    real_raw = pred_real[0] * action_std + action_mean
    black_raw = pred_black[0] * action_std + action_mean
    noise_raw = pred_noise[0] * action_std + action_mean

    print(f"\n  Real images action[0]:  {real_raw}")
    print(f"  Black images action[0]: {black_raw}")
    print(f"  Noise images action[0]: {noise_raw}")

    diff_black = np.abs(real_raw - black_raw)
    diff_noise = np.abs(real_raw - noise_raw)
    print(f"\n  |real - black|: {diff_black}  (sum={diff_black.sum():.6f})")
    print(f"  |real - noise|: {diff_noise}  (sum={diff_noise.sum():.6f})")

    vision_impact = diff_black.sum() / (np.abs(real_raw).sum() + 1e-10)
    print(f"\n  Vision impact ratio: {vision_impact:.4f}")
    if vision_impact < 0.1:
        print("  WARNING: Vision has very little impact on actions!")
        print("  The model may be ignoring images and relying on proprio/bias only.")
    elif vision_impact < 0.3:
        print("  Vision has moderate impact. May need more training for visual grounding.")
    else:
        print("  Vision has significant impact on actions. Good.")

    # =====================================================================
    # TEST 3: Proprio ablation
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 3: Proprio ablation (real proprio vs zeros)")
    print(f"{'='*70}")

    zero_proprio = torch.zeros_like(proprio_tensor)
    with torch.no_grad():
        pred_no_proprio = model.predict_action(
            images=[context_pil, wrist_pil],
            instruction=args.instruction,
            proprio_state=zero_proprio,
        )[0].float().cpu().numpy()

    no_proprio_raw = pred_no_proprio[0] * action_std + action_mean
    diff_proprio = np.abs(real_raw - no_proprio_raw)
    print(f"  Real proprio action[0]:  {real_raw}")
    print(f"  Zero proprio action[0]:  {no_proprio_raw}")
    print(f"  |real - zero|: {diff_proprio}  (sum={diff_proprio.sum():.6f})")

    proprio_impact = diff_proprio.sum() / (np.abs(real_raw).sum() + 1e-10)
    print(f"\n  Proprio impact ratio: {proprio_impact:.4f}")

    # =====================================================================
    # TEST 4: Feature norms through the pipeline
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 4: Feature norms through pipeline")
    print(f"{'='*70}")

    norms_list = []
    for i in range(min(5, args.num_samples)):
        sample = dataset[int(indices[i])]
        cp = tensor_to_pil(sample['context_image'])
        wp = tensor_to_pil(sample['wrist_image'])
        pt = sample['proprio_state'].unsqueeze(0).to(device, dtype=torch.bfloat16)

        norms = get_feature_norms(model, cp, wp, args.instruction, pt)
        norms_list.append(norms)

    print(f"\n  {'Stage':<20} {'Mean Norm':>12} {'Std Norm':>12}")
    print(f"  {'-'*44}")
    for key in norms_list[0].keys():
        vals = [n[key] for n in norms_list]
        print(f"  {key:<20} {np.mean(vals):>12.4f} {np.std(vals):>12.4f}")

    # Check for near-constant features (std ≈ 0)
    cond_vals = [n['condition'] for n in norms_list]
    if np.std(cond_vals) / (np.mean(cond_vals) + 1e-10) < 0.01:
        print("\n  WARNING: Conditioning vector norm is nearly constant across samples!")
        print("  The action expert may not be using observation information effectively.")

    # =====================================================================
    # TEST 5: Mode collapse check (multiple samples for same observation)
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 5: Mode collapse check (20 samples for same observation)")
    print(f"{'='*70}")

    sample = dataset[0]
    context_pil = tensor_to_pil(sample['context_image'])
    wrist_pil = tensor_to_pil(sample['wrist_image'])
    proprio_tensor = sample['proprio_state'].unsqueeze(0).to(device, dtype=torch.bfloat16)

    multi_preds = []
    for _ in range(20):
        with torch.no_grad():
            pred = model.predict_action(
                images=[context_pil, wrist_pil],
                instruction=args.instruction,
                proprio_state=proprio_tensor,
            )[0].float().cpu().numpy()
        multi_preds.append(pred[0])  # First step

    multi_preds = np.array(multi_preds)  # [20, 7]
    multi_raw = multi_preds * action_std + action_mean

    gt_normalized = sample['actions'].numpy()
    gt_raw = gt_normalized[0] * action_std + action_mean

    print(f"\n  Ground truth action[0]: {gt_raw}")
    print(f"  Pred mean (20 samples): {multi_raw.mean(axis=0)}")
    print(f"  Pred std  (20 samples): {multi_raw.std(axis=0)}")
    print(f"  Pred min  (20 samples): {multi_raw.min(axis=0)}")
    print(f"  Pred max  (20 samples): {multi_raw.max(axis=0)}")

    # Compare spread to GT action variation
    print(f"\n  Dataset action_std for reference: {action_std}")
    spread = multi_raw.std(axis=0)
    ratio = spread / (action_std + 1e-10)
    print(f"  Sampling spread / dataset_std:  {ratio}")
    print(f"  (<<0.1 = mode collapse, ~0.5-1.0 = reasonable diversity)")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}")

    issues = []

    # Check prediction quality
    mean_mae = all_errors.mean()
    print(f"\n1. Open-loop MAE: {mean_mae:.6f}")
    if mean_mae > action_std.mean() * 0.8:
        print(f"   POOR: Error is >{80}% of action std. Model predictions are nearly random.")
        issues.append("high_prediction_error")
    elif mean_mae > action_std.mean() * 0.5:
        print(f"   MODERATE: Error is ~50-80% of action std. Model has learned some structure.")
        issues.append("moderate_prediction_error")
    else:
        print(f"   GOOD: Error is <50% of action std.")

    # Check variance ratio
    avg_var_ratio = variance_ratio[:6].mean()  # Position/rotation dims
    print(f"\n2. Variance ratio (pred/gt, pos+rot dims): {avg_var_ratio:.4f}")
    if avg_var_ratio < 0.1:
        print("   MODE COLLAPSE: Model outputs are nearly constant. Predicting the mean action.")
        issues.append("mode_collapse")
    elif avg_var_ratio < 0.3:
        print("   LOW DIVERSITY: Model is under-confident, producing conservative actions.")
        issues.append("low_diversity")
    else:
        print("   OK: Reasonable action diversity.")

    # Vision check
    print(f"\n3. Vision impact: {vision_impact:.4f}")
    if vision_impact < 0.1:
        issues.append("vision_ignored")
        print("   BROKEN: Model ignores visual input.")
    elif vision_impact < 0.3:
        issues.append("weak_vision")
        print("   WEAK: Vision has limited influence. Needs more training.")

    # Proprio check
    print(f"\n4. Proprio impact: {proprio_impact:.4f}")
    if proprio_impact < 0.05:
        print("   MINIMAL: Proprio has little effect (may be fine if task is vision-only).")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    if "mode_collapse" in issues:
        print("""
- Mode collapse detected: the flow matching head outputs ~constant actions.
  Likely causes:
  a) Insufficient training (loss ~1.0 is still high for flow matching)
  b) The conditioning signal from action_expert is too weak/constant
  c) Too few flow matching ODE steps at inference

  Actions:
  1. Train longer — loss needs to reach ~0.3-0.5 for reasonable behavior
  2. Try reducing ODE steps from 100 to 10-20 (faster, sometimes better)
  3. Verify with overfit test: train on 5 demos to near-zero loss""")

    if "vision_ignored" in issues:
        print("""
- Vision pathway is broken or ignored.
  The model produces the same actions regardless of input images.

  Likely causes:
  a) VLM → projector bottleneck: single last-token embedding loses spatial info
  b) LoRA adapters not training effectively
  c) Projector output is near-constant (check feature norms above)

  Actions:
  1. Check if VLM embedding changes with different images (feature norms test)
  2. Try mean-pooling instead of last-token for VLM output
  3. Consider using cross-attention projector instead of MLP""")

    if "high_prediction_error" in issues and "mode_collapse" not in issues:
        print("""
- High prediction error with some diversity means the model is learning
  but needs more training or data.

  Actions:
  1. Continue training — loss was still declining at step 90k
  2. Try reducing learning rate (cosine decay should handle this)
  3. Increase training data diversity""")

    if not issues:
        print("""
- Model diagnostics look reasonable. The issue may be:
  a) Compounding errors in closed-loop (vs open-loop training)
  b) Distribution shift between dataset observations and sim
  c) Action chunking causing drift — try chunk_size=1 for debugging""")


if __name__ == "__main__":
    main()
