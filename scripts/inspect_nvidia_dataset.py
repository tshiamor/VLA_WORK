#!/usr/bin/env python3
"""
Inspect NVIDIA PhysicalAI-Robotics-Manipulation-Augmented HDF5 Dataset

Opens the HDF5 file and prints all group/dataset keys, shapes, and dtypes
for the first demo. Used to determine the exact key mapping needed for VLA training.

Usage:
    python scripts/inspect_nvidia_dataset.py
    python scripts/inspect_nvidia_dataset.py --hdf5_path /path/to/cosmos_dataset_1k.hdf5
    python scripts/inspect_nvidia_dataset.py --num_demos 5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def inspect_group(group, indent=0):
    """Recursively inspect an HDF5 group, printing keys, shapes, and dtypes."""
    prefix = "  " * indent
    for key in sorted(group.keys()):
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/ (group, {len(item)} items)")
            inspect_group(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}{key}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"{prefix}{key}: (unknown type)")


def inspect_dataset(hdf5_path: str, num_demos: int = 1):
    """Inspect the NVIDIA cosmos HDF5 dataset structure."""
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        print(f"ERROR: File not found: {hdf5_path}")
        return

    print(f"\n{'='*70}")
    print(f"Inspecting: {hdf5_path}")
    print(f"File size: {hdf5_path.stat().st_size / (1024**3):.2f} GB")
    print(f"{'='*70}")

    with h5py.File(hdf5_path, 'r') as f:
        # Top-level keys
        print(f"\nTop-level keys: {list(f.keys())}")

        # Check for 'data' group
        if 'data' in f:
            data_group = f['data']
            demo_names = sorted([k for k in data_group.keys() if k.startswith('demo_')])
            print(f"\nNumber of demos: {len(demo_names)}")

            # Inspect first N demos
            for i, demo_name in enumerate(demo_names[:num_demos]):
                demo = data_group[demo_name]
                print(f"\n{'─'*70}")
                print(f"Demo: {demo_name}")
                print(f"{'─'*70}")
                inspect_group(demo)

                # Print sample values for action shape understanding
                if 'actions' in demo:
                    actions = demo['actions']
                    print(f"\n  Actions summary:")
                    print(f"    Shape: {actions.shape}")
                    print(f"    First action: {actions[0][:10]}...")
                    print(f"    Min: {np.min(actions[:], axis=0)[:7]}")
                    print(f"    Max: {np.max(actions[:], axis=0)[:7]}")

                # Print image sample info
                if 'obs' in demo:
                    obs = demo['obs']
                    for key in sorted(obs.keys()):
                        if 'rgb' in key.lower() or 'image' in key.lower():
                            img_data = obs[key]
                            print(f"\n  Camera '{key}':")
                            print(f"    Shape: {img_data.shape}")
                            print(f"    Dtype: {img_data.dtype}")
                            print(f"    Value range: [{img_data[0].min()}, {img_data[0].max()}]")

            # Summary of episode lengths
            if len(demo_names) > 1:
                print(f"\n{'─'*70}")
                print("Episode length summary:")
                print(f"{'─'*70}")
                lengths = []
                for name in demo_names:
                    if 'actions' in data_group[name]:
                        lengths.append(data_group[name]['actions'].shape[0])
                if lengths:
                    print(f"  Total demos: {len(lengths)}")
                    print(f"  Min length: {min(lengths)}")
                    print(f"  Max length: {max(lengths)}")
                    print(f"  Mean length: {np.mean(lengths):.1f}")
                    print(f"  Total steps: {sum(lengths)}")
        else:
            print("\nNo 'data' group found. Top-level structure:")
            inspect_group(f)

    # Print suggested key mapping
    print(f"\n{'='*70}")
    print("Suggested key mapping for VLA training:")
    print(f"{'='*70}")
    print("""
Check the output above and create a key_mapping JSON file like:

{
    "context_rgb": "<table/agentview camera key from obs/>",
    "wrist_rgb": "<wrist/eye_in_hand camera key from obs/>",
    "joint_pos": "<joint position key from obs/>",
    "joint_vel": "<joint velocity key from obs/>",
    "eef_pos": "<end-effector position key from obs/>",
    "eef_quat": "<end-effector quaternion key from obs/>",
    "gripper_pos": "<gripper state key from obs/>",
    "actions": "<actions key (directly under demo, not obs)>"
}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect NVIDIA PhysicalAI HDF5 dataset structure"
    )
    parser.add_argument(
        "--hdf5_path", type=str,
        default="data/cosmos_dataset_1k.hdf5",
        help="Path to HDF5 file"
    )
    parser.add_argument(
        "--num_demos", type=int, default=2,
        help="Number of demos to inspect in detail"
    )
    args = parser.parse_args()

    inspect_dataset(args.hdf5_path, args.num_demos)


if __name__ == "__main__":
    main()
