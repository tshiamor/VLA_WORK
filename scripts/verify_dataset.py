#!/usr/bin/env python3
"""
Verify HDF5 Dataset for VLA Training

Quick script to verify your dataset is compatible with VLA training.

Usage:
    python scripts/verify_dataset.py --hdf5_path /path/to/demos.hdf5
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vla.data import HDF5VLADataset, verify_hdf5_dataset
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Verify HDF5 dataset for VLA")
    parser.add_argument("--hdf5_path", type=str, required=True,
                        help="Path to HDF5 demo file")
    parser.add_argument("--key_mapping_file", type=str, default=None,
                        help="Path to JSON key mapping file")
    parser.add_argument("--visualize", action="store_true",
                        help="Show sample images")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to check")
    args = parser.parse_args()

    # Load key mapping if provided
    key_mapping = None
    if args.key_mapping_file:
        with open(args.key_mapping_file, 'r') as f:
            key_mapping = json.load(f)
        print(f"Using key mapping from {args.key_mapping_file}")

    # Verify dataset structure
    print("\n" + "="*60)
    print("Step 1: Verify HDF5 Structure")
    print("="*60)
    stats = verify_hdf5_dataset(args.hdf5_path)

    # Load as VLA dataset
    print("\n" + "="*60)
    print("Step 2: Load as VLA Dataset")
    print("="*60)

    try:
        dataset_kwargs = dict(
            instruction="Pick up the card and place it in the target.",
            chunk_size=16,
            augment=False,
        )
        if key_mapping:
            dataset_kwargs["key_mapping"] = key_mapping
        dataset = HDF5VLADataset(args.hdf5_path, **dataset_kwargs)
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Action dim: {dataset.action_dim}")
        print(f"  - Proprio dim: {dataset.proprio_dim}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return 1

    # Test sample loading
    print("\n" + "="*60)
    print("Step 3: Test Sample Loading")
    print("="*60)

    for i in range(min(args.num_samples, len(dataset))):
        try:
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  context_image: {sample['context_image'].shape} ({sample['context_image'].dtype})")
            print(f"  wrist_image: {sample['wrist_image'].shape} ({sample['wrist_image'].dtype})")
            print(f"  proprio_state: {sample['proprio_state'].shape} ({sample['proprio_state'].dtype})")
            print(f"  actions: {sample['actions'].shape} ({sample['actions'].dtype})")
            print(f"  instruction: '{sample['instruction'][:50]}...'")
        except Exception as e:
            print(f"✗ Failed to load sample {i}: {e}")
            return 1

    print("\n✓ All samples loaded successfully!")

    # Print normalization stats
    print("\n" + "="*60)
    print("Step 4: Normalization Statistics")
    print("="*60)
    stats = dataset.get_stats()
    print(f"Action mean: {stats['action_mean']}")
    print(f"Action std:  {stats['action_std']}")
    print(f"Proprio mean: {stats['proprio_mean'][:5]}...")  # First 5
    print(f"Proprio std:  {stats['proprio_std'][:5]}...")

    # Visualize if requested
    if args.visualize:
        import matplotlib.pyplot as plt

        print("\n" + "="*60)
        print("Step 5: Visualize Samples")
        print("="*60)

        fig, axes = plt.subplots(2, args.num_samples, figsize=(4*args.num_samples, 8))

        for i in range(min(args.num_samples, len(dataset))):
            sample = dataset[i]

            # Context camera (top row)
            context_img = sample['context_image'].numpy().transpose(1, 2, 0)
            axes[0, i].imshow(context_img)
            axes[0, i].set_title(f"Context {i}")
            axes[0, i].axis('off')

            # Wrist camera (bottom row)
            wrist_img = sample['wrist_image'].numpy().transpose(1, 2, 0)
            axes[1, i].imshow(wrist_img)
            axes[1, i].set_title(f"Wrist {i}")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig("/tmp/vla_dataset_samples.png")
        print(f"Saved visualization to /tmp/vla_dataset_samples.png")
        plt.show()

    print("\n" + "="*60)
    print("Dataset Verification Complete!")
    print("="*60)
    print(f"\nYour dataset is ready for VLA training.")
    print(f"\nTo train, run:")
    print(f"  python scripts/train.py \\")
    print(f"    --hdf5_path {args.hdf5_path} \\")
    print(f"    --instruction 'Pick up the card and place it in the target.' \\")
    print(f"    --action_dim {dataset.action_dim} \\")
    print(f"    --epochs 50")

    return 0


if __name__ == "__main__":
    sys.exit(main())
