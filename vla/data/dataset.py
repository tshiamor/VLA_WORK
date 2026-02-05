"""
Dataset classes for VLA training

Handles loading and processing of robot demonstration data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import pickle
from PIL import Image
import h5py


@dataclass
class DemoEpisode:
    """Single demonstration episode data."""
    # Images: List of (context_img, wrist_img) tuples
    images: List[Tuple[np.ndarray, np.ndarray]]
    # Actions: [T, action_dim]
    actions: np.ndarray
    # Proprioceptive states: [T, proprio_dim]
    proprio_states: np.ndarray
    # Language instruction
    instruction: str
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.actions)


class VLADataset(Dataset):
    """
    PyTorch Dataset for VLA training.

    Handles demonstration data with:
    - Multi-camera images
    - Action sequences (with chunking)
    - Proprioceptive states
    - Language instructions
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        chunk_size: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        action_dim: int = 7,
        proprio_dim: int = 14,
        augment: bool = True,
        max_episodes: Optional[int] = None,
    ):
        """
        Initialize VLA dataset.

        Args:
            data_dir: Directory containing demonstration data
            chunk_size: Number of action steps per chunk
            image_size: Target image size (H, W)
            action_dim: Dimension of action space
            proprio_dim: Dimension of proprioceptive state
            augment: Whether to apply data augmentation
            max_episodes: Maximum number of episodes to load
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.augment = augment

        # Load episode index
        self.episodes: List[DemoEpisode] = []
        self._load_episodes(max_episodes)

        # Build sample index (episode_idx, timestep)
        self.sample_index: List[Tuple[int, int]] = []
        self._build_sample_index()

        # Compute action statistics for normalization
        self.action_mean, self.action_std = self._compute_action_stats()

    def _load_episodes(self, max_episodes: Optional[int] = None) -> None:
        """Load all episodes from data directory."""
        # Support multiple data formats
        episode_files = []

        # HDF5 files
        episode_files.extend(self.data_dir.glob("*.h5"))
        episode_files.extend(self.data_dir.glob("*.hdf5"))

        # Pickle files
        episode_files.extend(self.data_dir.glob("*.pkl"))

        # JSON metadata files (with separate image folders)
        episode_files.extend(self.data_dir.glob("*/episode.json"))

        episode_files = sorted(episode_files)
        if max_episodes:
            episode_files = episode_files[:max_episodes]

        for ep_file in episode_files:
            try:
                if ep_file.suffix in ['.h5', '.hdf5']:
                    episode = self._load_hdf5_episode(ep_file)
                elif ep_file.suffix == '.pkl':
                    episode = self._load_pickle_episode(ep_file)
                elif ep_file.name == 'episode.json':
                    episode = self._load_json_episode(ep_file)
                else:
                    continue

                if episode is not None:
                    self.episodes.append(episode)
            except Exception as e:
                print(f"Warning: Failed to load {ep_file}: {e}")

        print(f"Loaded {len(self.episodes)} episodes")

    def _load_hdf5_episode(self, path: Path) -> Optional[DemoEpisode]:
        """Load episode from HDF5 file."""
        with h5py.File(path, 'r') as f:
            # Load images
            context_imgs = f['context_images'][:]
            wrist_imgs = f['wrist_images'][:]
            images = list(zip(context_imgs, wrist_imgs))

            # Load actions and states
            actions = f['actions'][:]
            proprio_states = f['proprio_states'][:]

            # Load instruction
            instruction = f.attrs.get('instruction', 'Perform the task.')
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')

            # Metadata
            metadata = dict(f.attrs)

            return DemoEpisode(
                images=images,
                actions=actions,
                proprio_states=proprio_states,
                instruction=instruction,
                metadata=metadata,
            )

    def _load_pickle_episode(self, path: Path) -> Optional[DemoEpisode]:
        """Load episode from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        return DemoEpisode(
            images=data['images'],
            actions=np.array(data['actions']),
            proprio_states=np.array(data['proprio_states']),
            instruction=data.get('instruction', 'Perform the task.'),
            metadata=data.get('metadata', {}),
        )

    def _load_json_episode(self, path: Path) -> Optional[DemoEpisode]:
        """Load episode from JSON metadata + image folder."""
        ep_dir = path.parent

        with open(path, 'r') as f:
            metadata = json.load(f)

        # Load images
        images = []
        for i in range(metadata['length']):
            context_path = ep_dir / f"context_{i:06d}.png"
            wrist_path = ep_dir / f"wrist_{i:06d}.png"

            if context_path.exists() and wrist_path.exists():
                context_img = np.array(Image.open(context_path))
                wrist_img = np.array(Image.open(wrist_path))
                images.append((context_img, wrist_img))

        return DemoEpisode(
            images=images,
            actions=np.array(metadata['actions']),
            proprio_states=np.array(metadata['proprio_states']),
            instruction=metadata.get('instruction', 'Perform the task.'),
            metadata=metadata,
        )

    def _build_sample_index(self) -> None:
        """Build index of valid (episode, timestep) samples."""
        for ep_idx, episode in enumerate(self.episodes):
            # Can sample from any timestep that allows full chunk
            max_start = episode.length - self.chunk_size
            for t in range(max(1, max_start + 1)):
                self.sample_index.append((ep_idx, t))

        print(f"Built sample index with {len(self.sample_index)} samples")

    def _compute_action_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std of actions for normalization."""
        all_actions = []
        for episode in self.episodes:
            all_actions.append(episode.actions)

        if not all_actions:
            return np.zeros(self.action_dim), np.ones(self.action_dim)

        all_actions = np.concatenate(all_actions, axis=0)
        mean = all_actions.mean(axis=0)
        std = all_actions.std(axis=0) + 1e-8

        return mean, std

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
        ep_idx, t = self.sample_index[idx]
        episode = self.episodes[ep_idx]

        # Get current observation
        context_img, wrist_img = episode.images[t]
        proprio_state = episode.proprio_states[t]

        # Get action chunk
        end_t = min(t + self.chunk_size, episode.length)
        actions = episode.actions[t:end_t]

        # Pad if necessary
        if len(actions) < self.chunk_size:
            pad_length = self.chunk_size - len(actions)
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (pad_length, 1))
            ], axis=0)

        # Normalize actions
        actions = (actions - self.action_mean) / self.action_std

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
            'instruction': episode.instruction,
        }

    def _process_image(self, img: np.ndarray) -> np.ndarray:
        """Process image: resize and normalize."""
        # Resize if needed
        if img.shape[:2] != self.image_size:
            img = np.array(Image.fromarray(img).resize(
                (self.image_size[1], self.image_size[0]),
                Image.BILINEAR
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


class VLADemoDataset(VLADataset):
    """
    Extended dataset for collecting demonstrations from Isaac Lab.

    Provides utilities for recording and saving new demonstrations.
    """

    def __init__(self, data_dir: Union[str, Path], **kwargs):
        # Create data directory if it doesn't exist
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty if no existing data
        self.episodes = []
        self.sample_index = []
        self.action_mean = np.zeros(kwargs.get('action_dim', 7))
        self.action_std = np.ones(kwargs.get('action_dim', 7))

        self.chunk_size = kwargs.get('chunk_size', 16)
        self.image_size = kwargs.get('image_size', (224, 224))
        self.action_dim = kwargs.get('action_dim', 7)
        self.proprio_dim = kwargs.get('proprio_dim', 14)
        self.augment = kwargs.get('augment', False)

        # Recording state
        self._recording = False
        self._current_episode: Optional[Dict] = None

    def start_recording(self, instruction: str) -> None:
        """Start recording a new demonstration."""
        self._recording = True
        self._current_episode = {
            'images': [],
            'actions': [],
            'proprio_states': [],
            'instruction': instruction,
        }

    def record_step(
        self,
        context_img: np.ndarray,
        wrist_img: np.ndarray,
        action: np.ndarray,
        proprio_state: np.ndarray,
    ) -> None:
        """Record a single timestep."""
        if not self._recording:
            return

        self._current_episode['images'].append((context_img.copy(), wrist_img.copy()))
        self._current_episode['actions'].append(action.copy())
        self._current_episode['proprio_states'].append(proprio_state.copy())

    def stop_recording(self, save: bool = True) -> Optional[DemoEpisode]:
        """Stop recording and optionally save the episode."""
        if not self._recording:
            return None

        self._recording = False

        if len(self._current_episode['actions']) < self.chunk_size:
            print("Episode too short, discarding")
            self._current_episode = None
            return None

        episode = DemoEpisode(
            images=self._current_episode['images'],
            actions=np.array(self._current_episode['actions']),
            proprio_states=np.array(self._current_episode['proprio_states']),
            instruction=self._current_episode['instruction'],
        )

        if save:
            self._save_episode(episode)
            self.episodes.append(episode)
            self._build_sample_index()
            self.action_mean, self.action_std = self._compute_action_stats()

        self._current_episode = None
        return episode

    def _save_episode(self, episode: DemoEpisode) -> None:
        """Save episode to disk."""
        ep_idx = len(self.episodes)
        ep_path = self.data_dir / f"episode_{ep_idx:06d}.h5"

        with h5py.File(ep_path, 'w') as f:
            # Save images
            context_imgs = np.stack([img[0] for img in episode.images])
            wrist_imgs = np.stack([img[1] for img in episode.images])
            f.create_dataset('context_images', data=context_imgs, compression='gzip')
            f.create_dataset('wrist_images', data=wrist_imgs, compression='gzip')

            # Save actions and states
            f.create_dataset('actions', data=episode.actions)
            f.create_dataset('proprio_states', data=episode.proprio_states)

            # Save instruction as attribute
            f.attrs['instruction'] = episode.instruction

        print(f"Saved episode to {ep_path}")


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """Create a DataLoader for VLA training."""
    dataset = VLADataset(data_dir, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
