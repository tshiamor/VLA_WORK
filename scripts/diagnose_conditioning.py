#!/usr/bin/env python3
"""
Diagnose why the VLA can't overfit: is the condition vector informative?

Tests:
1. Are VLM embeddings identical across samples? (frozen VLM, same task)
2. Does the action expert produce different conditions for different proprios?
3. Can the flow head overfit when given unique per-sample conditions?
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla.models import create_vla_model
from vla.data import HDF5VLADataset
from vla.models.flow_matching import FlowMatchingConfig, FlowMatchingActionHead


def test_vlm_embedding_variance(model, dataset, device, n_samples=20):
    """Test 1: Do VLM embeddings vary across samples?"""
    print("\n" + "="*60)
    print("TEST 1: VLM embedding variance across samples")
    print("="*60)

    model.eval()
    embeddings = []

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        ctx = sample['context_image'].unsqueeze(0)
        wrist = sample['wrist_image'].unsqueeze(0)
        instruction = [sample['instruction']]

        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            vlm_out = model.vision_encoder(
                context_images=ctx,
                wrist_images=wrist,
                instructions=instruction,
                return_last_hidden_state=True,
            )
        emb = vlm_out['last_hidden_state'].float().cpu()  # [1, 1, 3584]
        embeddings.append(emb.squeeze())

    embeddings = torch.stack(embeddings)  # [N, 3584]

    # Compute pairwise cosine similarity
    normed = F.normalize(embeddings, dim=-1)
    sim_matrix = normed @ normed.T
    off_diag = sim_matrix[~torch.eye(n_samples, dtype=bool)]

    # Compute variance per dimension
    per_dim_std = embeddings.std(dim=0)

    print(f"  Samples: {n_samples}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Mean cosine similarity: {off_diag.mean():.6f}")
    print(f"  Min cosine similarity:  {off_diag.min():.6f}")
    print(f"  Max cosine similarity:  {off_diag.max():.6f}")
    print(f"  Mean per-dim std:       {per_dim_std.mean():.6f}")
    print(f"  Max per-dim std:        {per_dim_std.max():.6f}")
    print(f"  Dims with std > 0.01:   {(per_dim_std > 0.01).sum()}/{embeddings.shape[1]}")

    if off_diag.mean() > 0.99:
        print("  PROBLEM: VLM embeddings are nearly identical across samples!")
    else:
        print("  OK: VLM embeddings vary across samples.")

    return embeddings


def test_condition_variance(model, dataset, device, n_samples=20):
    """Test 2: Does the condition vector vary across samples?"""
    print("\n" + "="*60)
    print("TEST 2: Condition vector variance across samples")
    print("="*60)

    model.eval()
    conditions = []
    proprios = []

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        ctx = sample['context_image'].unsqueeze(0)
        wrist = sample['wrist_image'].unsqueeze(0)
        proprio = sample['proprio_state'].unsqueeze(0).to(device)
        instruction = [sample['instruction']]

        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            vlm_out = model.vision_encoder(
                context_images=ctx,
                wrist_images=wrist,
                instructions=instruction,
                return_last_hidden_state=True,
            )
            vlm_emb = vlm_out.get('last_hidden_state', vlm_out.get('pooled_output'))
            projected = model.projector(vlm_emb)
            expert_out = model.action_expert(
                vlm_embeddings=projected,
                proprio_state=proprio,
            )
            cond = expert_out['condition'].float().cpu().squeeze()

        conditions.append(cond)
        proprios.append(sample['proprio_state'])

    conditions = torch.stack(conditions)  # [N, 512]
    proprios = torch.stack(proprios)      # [N, 27]

    # Condition similarity
    normed_c = F.normalize(conditions, dim=-1)
    sim_c = normed_c @ normed_c.T
    off_diag_c = sim_c[~torch.eye(n_samples, dtype=bool)]

    # Proprio similarity
    normed_p = F.normalize(proprios, dim=-1)
    sim_p = normed_p @ normed_p.T
    off_diag_p = sim_p[~torch.eye(n_samples, dtype=bool)]

    per_dim_std_c = conditions.std(dim=0)
    per_dim_std_p = proprios.std(dim=0)

    print(f"  Proprio cosine sim:    mean={off_diag_p.mean():.4f}, min={off_diag_p.min():.4f}")
    print(f"  Proprio per-dim std:   mean={per_dim_std_p.mean():.4f}")
    print(f"  Condition cosine sim:  mean={off_diag_c.mean():.6f}, min={off_diag_c.min():.6f}")
    print(f"  Condition per-dim std: mean={per_dim_std_c.mean():.6f}")
    print(f"  Cond dims with std > 0.01: {(per_dim_std_c > 0.01).sum()}/{conditions.shape[1]}")

    if off_diag_c.mean() > 0.99:
        print("  PROBLEM: Condition vectors are nearly identical!")
        print("  The action expert is NOT transmitting proprio information.")
    elif off_diag_c.mean() > 0.95:
        print("  WARNING: Condition vectors are very similar.")
        print("  Proprio info may be too diluted in the condition.")
    else:
        print("  OK: Condition vectors vary across samples.")


def test_flow_head_isolation(device, n_samples=100, n_steps=3000):
    """Test 3: Can the flow head overfit with unique per-sample conditions?"""
    print("\n" + "="*60)
    print("TEST 3: Flow head overfitting in isolation")
    print("="*60)

    config = FlowMatchingConfig(action_dim=7, hidden_dim=512, chunk_size=16)
    flow_head = FlowMatchingActionHead(config).to(device)

    # Create synthetic data: unique condition per sample, fixed target actions
    torch.manual_seed(42)
    conditions = torch.randn(n_samples, 512, device=device)  # unique per sample
    target_actions = torch.randn(n_samples, 16, 7, device=device) * 0.5  # target actions

    optimizer = torch.optim.AdamW(flow_head.parameters(), lr=1e-3)

    print(f"  Samples: {n_samples}, Steps: {n_steps}")
    print(f"  Flow head params: {sum(p.numel() for p in flow_head.parameters()):,}")

    losses = []
    for step in range(n_steps):
        # Random batch
        idx = torch.randint(0, n_samples, (8,))
        cond_batch = conditions[idx]
        action_batch = target_actions[idx]

        loss, _ = flow_head.compute_loss(action_batch, cond_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_head.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            avg = np.mean(losses[-500:])
            print(f"  Step {step+1:5d} | Loss: {loss.item():.4f} | Avg(500): {avg:.4f}")

    final_avg = np.mean(losses[-100:])
    print(f"\n  Final avg loss (last 100): {final_avg:.4f}")

    if final_avg < 0.1:
        print("  PASS: Flow head CAN overfit with unique conditions.")
        print("  The issue is in the conditioning pipeline, not the flow head.")
    elif final_avg < 0.5:
        print("  PARTIAL: Flow head partially overfits.")
    else:
        print("  FAIL: Flow head CANNOT overfit even with unique conditions.")
        print("  The issue is in the flow matching architecture itself.")


def main():
    device = torch.device("cuda:0")

    # Load key mapping
    km_path = PROJECT_ROOT / "configs/key_mappings/nvidia_cosmos.json"
    key_mapping = None
    if km_path.exists():
        with open(km_path) as f:
            key_mapping = json.load(f)

    # Test 3 first (no VLM needed, fast)
    test_flow_head_isolation(device)

    # Load dataset (5 demos)
    print("\nLoading dataset...")
    hdf5_path = str(PROJECT_ROOT / "data/cosmos_dataset_1k.hdf5")
    ds_kwargs = dict(instruction="Stack the cubes.", chunk_size=16, augment=False, max_demos=5)
    if key_mapping:
        ds_kwargs["key_mapping"] = key_mapping
    dataset = HDF5VLADataset(hdf5_path, **ds_kwargs)

    # Load model
    print("Loading VLA model...")
    model = create_vla_model(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        action_dim=7, chunk_size=16, hidden_dim=512, proprio_dim=27,
        freeze_vlm=True, use_lora=True, device_map="auto",
    )
    model.projector.to(device)
    model.action_expert.to(device)
    model.flow_head.to(device)

    test_vlm_embedding_variance(model, dataset, device)
    test_condition_variance(model, dataset, device)

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
