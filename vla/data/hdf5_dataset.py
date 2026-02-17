"""
HDF5 Dataset Loader for VLA Training

Specialized loader for Isaac Lab demonstration HDF5 files with the structure:
- data/demo_N/obs/table_rgb, wrist_rgb (cameras)
- data/demo_N/obs/joint_pos, joint_vel, eef_pos, eef_quat, gripper_pos (proprioception)
- data/demo_N/actions (7-DOF actions)

Compatible with mcx_card_demos_vla_224.hdf5 and similar datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import h5py


class HDF5VLADataset(Dataset):
    """
    PyTorch Dataset for VLA training from consolidated HDF5 files.

    Supports the Isaac Lab demo format with all demos in a single HDF5 file.
    Also supports external datasets (e.g. NVIDIA cosmos) via key_mapping.
    """

    def __init__(
        self,
        hdf5_path: Union[str, Path],
        instruction: str = "Pick up the object and place it in the target location.",
        chunk_size: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        context_camera_key: str = "table_rgb",
        wrist_camera_key: str = "wrist_rgb",
        action_key: str = "actions",
        augment: bool = True,
        max_demos: Optional[int] = None,
        proprio_keys: Optional[List[str]] = None,
        key_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize HDF5 VLA dataset.

        Args:
            hdf5_path: Path to HDF5 file containing all demos
            instruction: Language instruction for the task
            chunk_size: Number of action steps per chunk
            image_size: Target image size (H, W)
            context_camera_key: Key for context/table camera in obs group
            wrist_camera_key: Key for wrist camera in obs group
            action_key: Key for actions dataset
            augment: Whether to apply data augmentation
            max_demos: Maximum number of demos to load (None = all)
            proprio_keys: List of proprioception keys to include
            key_mapping: Optional dict mapping canonical names to dataset-specific
                keys. Supported canonical names:
                - "context_rgb": maps to context camera key in obs/
                - "wrist_rgb": maps to wrist camera key in obs/
                - "joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_pos":
                  maps to proprioception keys in obs/
                - "actions": maps to action key (under demo, not obs)
                When provided, overrides context_camera_key, wrist_camera_key,
                action_key, and proprio_keys with the mapped values.
        """
        self.hdf5_path = Path(hdf5_path)
        self.instruction = instruction
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.augment = augment
        self.key_mapping = key_mapping

        # Apply key_mapping if provided
        if key_mapping:
            self.context_key = key_mapping.get("context_rgb", context_camera_key)
            self.wrist_key = key_mapping.get("wrist_rgb", wrist_camera_key)
            self.action_key = key_mapping.get("actions", action_key)

            # Build proprio_keys from mapping — use mapped values for canonical names
            canonical_proprio = ["joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_pos"]
            self.proprio_keys = [
                key_mapping[k] for k in canonical_proprio if k in key_mapping
            ]
            # If no proprio keys were mapped, fall back to defaults
            if not self.proprio_keys:
                self.proprio_keys = proprio_keys or canonical_proprio
        else:
            self.context_key = context_camera_key
            self.wrist_key = wrist_camera_key
            self.action_key = action_key
            # Default proprioception keys
            self.proprio_keys = proprio_keys or [
                "joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_pos"
            ]

        # Load metadata and build index
        self.demo_info: List[Dict] = []
        self.sample_index: List[Tuple[int, int]] = []

        self._scan_hdf5(max_demos)
        self._build_sample_index()

        # Compute action statistics
        self.action_mean, self.action_std = self._compute_action_stats()
        self.action_dim = self.action_mean.shape[0]

        # Compute proprio statistics
        self.proprio_mean, self.proprio_std = self._compute_proprio_stats()
        self.proprio_dim = self.proprio_mean.shape[0]

        print(f"Loaded {len(self.demo_info)} demos, {len(self.sample_index)} samples")
        print(f"Action dim: {self.action_dim}, Proprio dim: {self.proprio_dim}")

    def _scan_hdf5(self, max_demos: Optional[int] = None) -> None:
        """Scan HDF5 file to find all demos and their lengths."""
        with h5py.File(self.hdf5_path, 'r') as f:
            data_group = f['data']
            demo_names = sorted([k for k in data_group.keys() if k.startswith('demo_')])

            if max_demos:
                demo_names = demo_names[:max_demos]

            # Probe proprio key dimensions from all demos (some may have keys others lack)
            self._proprio_key_dims: Dict[str, int] = {}
            for demo_name in demo_names:
                obs = data_group[demo_name]['obs']
                for key in self.proprio_keys:
                    if key in obs and key not in self._proprio_key_dims:
                        self._proprio_key_dims[key] = obs[key].shape[-1]
                if len(self._proprio_key_dims) == len(self.proprio_keys):
                    break  # Found all key dimensions

            for demo_name in demo_names:
                demo = data_group[demo_name]

                # Get episode length from actions
                actions = demo[self.action_key]
                length = actions.shape[0]

                # Verify camera data exists
                obs = demo['obs']
                has_context = self.context_key in obs
                has_wrist = self.wrist_key in obs

                if not has_context or not has_wrist:
                    print(f"Warning: {demo_name} missing camera data, skipping")
                    continue

                self.demo_info.append({
                    'name': demo_name,
                    'length': length,
                })

    def _build_sample_index(self) -> None:
        """Build index of valid (demo_idx, timestep) samples."""
        for demo_idx, info in enumerate(self.demo_info):
            # Can sample from any timestep that allows full action chunk
            max_start = info['length'] - self.chunk_size
            for t in range(max(1, max_start + 1)):
                self.sample_index.append((demo_idx, t))

    def _compute_action_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std of actions for normalization."""
        all_actions = []

        with h5py.File(self.hdf5_path, 'r') as f:
            data_group = f['data']
            for info in self.demo_info:
                actions = data_group[info['name']][self.action_key][:]
                all_actions.append(actions)

        if not all_actions:
            return np.zeros(7), np.ones(7)

        all_actions = np.concatenate(all_actions, axis=0)
        mean = all_actions.mean(axis=0)
        std = all_actions.std(axis=0) + 1e-8

        return mean.astype(np.float32), std.astype(np.float32)

    def _compute_proprio_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std of proprioception for normalization."""
        all_proprio = []

        with h5py.File(self.hdf5_path, 'r') as f:
            data_group = f['data']
            for info in self.demo_info:
                obs = data_group[info['name']]['obs']
                proprio = self._extract_proprio(obs, slice(None))
                all_proprio.append(proprio)

        if not all_proprio:
            return np.zeros(14), np.ones(14)

        all_proprio = np.concatenate(all_proprio, axis=0)
        mean = all_proprio.mean(axis=0)
        std = all_proprio.std(axis=0) + 1e-8

        return mean.astype(np.float32), std.astype(np.float32)

    def _extract_proprio(self, obs_group, idx) -> np.ndarray:
        """Extract and concatenate proprioceptive features.

        Missing keys are zero-padded to maintain a consistent proprio dimension
        across demos that may have different observation sets.
        """
        proprio_parts = []

        for key in self.proprio_keys:
            if key in obs_group:
                data = obs_group[key][idx]
                if data.ndim == 1:
                    data = data.reshape(-1, data.shape[0]) if isinstance(idx, slice) else data.reshape(1, -1)
                proprio_parts.append(data)
            elif key in self._proprio_key_dims:
                # Key missing in this demo — zero-pad with the expected dimension
                dim = self._proprio_key_dims[key]
                if isinstance(idx, slice):
                    n_steps = obs_group[self.context_key].shape[0]
                    proprio_parts.append(np.zeros((n_steps, dim), dtype=np.float32))
                else:
                    proprio_parts.append(np.zeros((1, dim), dtype=np.float32))

        if not proprio_parts:
            total_dim = sum(self._proprio_key_dims.values()) if self._proprio_key_dims else 14
            if isinstance(idx, slice):
                return np.zeros((obs_group[self.context_key].shape[0], total_dim), dtype=np.float32)
            return np.zeros(total_dim, dtype=np.float32)

        result = np.concatenate(proprio_parts, axis=-1)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        return result.squeeze() if not isinstance(idx, slice) else result

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dictionary containing:
            - 'context_image': [3, H, W] float tensor
            - 'wrist_image': [3, H, W] float tensor
            - 'proprio_state': [proprio_dim] float tensor
            - 'actions': [chunk_size, action_dim] float tensor
            - 'instruction': str
        """
        demo_idx, t = self.sample_index[idx]
        info = self.demo_info[demo_idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            demo = f['data'][info['name']]
            obs = demo['obs']

            # Get current camera observations
            context_img = obs[self.context_key][t]
            wrist_img = obs[self.wrist_key][t]

            # Get proprioception
            proprio_state = self._extract_proprio(obs, t)

            # Get action chunk
            end_t = min(t + self.chunk_size, info['length'])
            actions = demo[self.action_key][t:end_t]

        # Pad actions if necessary
        if len(actions) < self.chunk_size:
            pad_length = self.chunk_size - len(actions)
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (pad_length, 1))
            ], axis=0)

        # Normalize actions
        actions = (actions - self.action_mean) / self.action_std

        # Normalize proprioception
        proprio_state = (proprio_state - self.proprio_mean) / self.proprio_std

        # Process images
        context_img = self._process_image(context_img)
        wrist_img = self._process_image(wrist_img)

        # Apply augmentation
        if self.augment:
            context_img, wrist_img = self._augment_images(context_img, wrist_img)

        return {
            'context_image': torch.tensor(context_img, dtype=torch.float32),
            'wrist_image': torch.tensor(wrist_img, dtype=torch.float32),
            'proprio_state': torch.tensor(proprio_state, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.float32),
            'instruction': self.instruction,
        }

    def _process_image(self, img: np.ndarray) -> np.ndarray:
        """Process image: resize and normalize."""
        from PIL import Image as PILImage

        # Resize if needed
        if img.shape[:2] != self.image_size:
            img = np.array(PILImage.fromarray(img).resize(
                (self.image_size[1], self.image_size[0]),
                PILImage.BILINEAR
            ))

        # Convert to float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Channel first: [H, W, C] -> [C, H, W]
        img = np.transpose(img, (2, 0, 1))

        return img

    def _augment_images(
        self,
        context_img: np.ndarray,
        wrist_img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply consistent augmentation to both images."""
        # Random brightness/contrast
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)

            context_img = np.clip(contrast * (context_img - 0.5) + 0.5 + brightness - 1, 0, 1)
            wrist_img = np.clip(contrast * (wrist_img - 0.5) + 0.5 + brightness - 1, 0, 1)

        # Random color jitter
        if np.random.random() < 0.3:
            for c in range(3):
                jitter = np.random.uniform(0.95, 1.05)
                context_img[c] = np.clip(context_img[c] * jitter, 0, 1)
                wrist_img[c] = np.clip(wrist_img[c] * jitter, 0, 1)

        return context_img, wrist_img

    def get_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization statistics."""
        return {
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'proprio_mean': self.proprio_mean,
            'proprio_std': self.proprio_std,
        }


def create_hdf5_dataloader(
    hdf5_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """Create a DataLoader for VLA training from HDF5 file."""
    dataset = HDF5VLADataset(hdf5_path, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# Convenience function to verify dataset
def verify_hdf5_dataset(hdf5_path: str) -> Dict:
    """
    Verify HDF5 dataset structure and print summary.

    Returns dict with dataset statistics.
    """
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f['data']
        demo_names = [k for k in data_group.keys() if k.startswith('demo_')]

        total_steps = 0
        action_dims = set()

        print(f"\n{'='*60}")
        print(f"HDF5 Dataset: {hdf5_path}")
        print(f"{'='*60}")
        print(f"Number of demos: {len(demo_names)}")

        if demo_names:
            demo = data_group[demo_names[0]]
            obs = demo['obs']

            print(f"\nObservation keys:")
            for key in obs.keys():
                shape = obs[key].shape
                dtype = obs[key].dtype
                print(f"  - {key}: {shape} ({dtype})")

            print(f"\nAction shape: {demo['actions'].shape}")

            for name in demo_names:
                total_steps += data_group[name]['actions'].shape[0]
                action_dims.add(data_group[name]['actions'].shape[1])

        print(f"\nTotal steps: {total_steps:,}")
        print(f"Action dims: {action_dims}")
        print(f"{'='*60}\n")

        return {
            'num_demos': len(demo_names),
            'total_steps': total_steps,
            'action_dims': list(action_dims),
        }
