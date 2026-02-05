"""
Preprocessing utilities for VLA data
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class ActionNormalizerConfig:
    """Configuration for action normalization."""
    action_dim: int = 7
    normalize: bool = True
    clip_range: float = 5.0  # Clip to [-clip_range, clip_range] after normalization


class ActionNormalizer:
    """
    Normalizes and denormalizes robot actions.

    Supports:
    - Z-score normalization (zero mean, unit variance)
    - Min-max normalization
    - Per-dimension statistics
    """

    def __init__(
        self,
        action_dim: int = 7,
        normalize: bool = True,
        clip_range: float = 5.0,
    ):
        self.action_dim = action_dim
        self.normalize = normalize
        self.clip_range = clip_range

        # Statistics (initialized to identity transform)
        self.mean = np.zeros(action_dim)
        self.std = np.ones(action_dim)
        self.min_val = np.zeros(action_dim)
        self.max_val = np.ones(action_dim)

        self._fitted = False

    def fit(self, actions: np.ndarray) -> 'ActionNormalizer':
        """
        Compute normalization statistics from data.

        Args:
            actions: Action data [N, action_dim] or list of [T, action_dim]

        Returns:
            self for chaining
        """
        if isinstance(actions, list):
            actions = np.concatenate(actions, axis=0)

        self.mean = actions.mean(axis=0)
        self.std = actions.std(axis=0) + 1e-8
        self.min_val = actions.min(axis=0)
        self.max_val = actions.max(axis=0)

        self._fitted = True
        return self

    def transform(
        self,
        actions: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize actions.

        Args:
            actions: Raw actions [..., action_dim]

        Returns:
            Normalized actions
        """
        if not self.normalize:
            return actions

        is_tensor = isinstance(actions, torch.Tensor)
        if is_tensor:
            device = actions.device
            dtype = actions.dtype
            actions = actions.cpu().numpy()

        # Z-score normalization
        normalized = (actions - self.mean) / self.std

        # Clip to range
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        if is_tensor:
            normalized = torch.tensor(normalized, dtype=dtype, device=device)

        return normalized

    def inverse_transform(
        self,
        actions: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize actions back to original scale.

        Args:
            actions: Normalized actions [..., action_dim]

        Returns:
            Denormalized actions
        """
        if not self.normalize:
            return actions

        is_tensor = isinstance(actions, torch.Tensor)
        if is_tensor:
            device = actions.device
            dtype = actions.dtype
            actions = actions.cpu().numpy()

        # Inverse z-score
        denormalized = actions * self.std + self.mean

        if is_tensor:
            denormalized = torch.tensor(denormalized, dtype=dtype, device=device)

        return denormalized

    def save(self, path: str) -> None:
        """Save normalizer statistics."""
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            min_val=self.min_val,
            max_val=self.max_val,
        )

    def load(self, path: str) -> 'ActionNormalizer':
        """Load normalizer statistics."""
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.min_val = data['min_val']
        self.max_val = data['max_val']
        self._fitted = True
        return self


class ImagePreprocessor:
    """
    Preprocesses images for VLA models.

    Handles:
    - Resizing to target size
    - Normalization (ImageNet or custom stats)
    - Channel ordering (CHW vs HWC)
    """

    # ImageNet normalization stats
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: str = "imagenet",  # "imagenet", "unit", or "none"
        channel_first: bool = True,
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.channel_first = channel_first

        # Set normalization stats
        if normalize == "imagenet":
            self.mean = self.IMAGENET_MEAN
            self.std = self.IMAGENET_STD
        elif normalize == "unit":
            self.mean = np.array([0.5, 0.5, 0.5])
            self.std = np.array([0.5, 0.5, 0.5])
        else:
            self.mean = np.array([0.0, 0.0, 0.0])
            self.std = np.array([1.0, 1.0, 1.0])

    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Process a single image."""
        return self.transform(image)

    def transform(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform image for model input.

        Args:
            image: Input image [H, W, 3] uint8 or float

        Returns:
            Processed image [3, H, W] or [H, W, 3] float32
        """
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            device = image.device
            image = image.cpu().numpy()

        # Ensure float [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image / 255.0

        # Resize if needed
        if image.shape[:2] != self.target_size:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray((image * 255).astype(np.uint8))
            pil_img = pil_img.resize(
                (self.target_size[1], self.target_size[0]),
                PILImage.BILINEAR
            )
            image = np.array(pil_img).astype(np.float32) / 255.0

        # Normalize
        if self.normalize != "none":
            image = (image - self.mean) / self.std

        # Channel ordering
        if self.channel_first:
            image = np.transpose(image, (2, 0, 1))

        if is_tensor:
            image = torch.tensor(image, dtype=torch.float32, device=device)

        return image

    def inverse_transform(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Convert processed image back to displayable format.

        Args:
            image: Processed image [3, H, W] or [H, W, 3]

        Returns:
            Image [H, W, 3] uint8
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Handle channel ordering
        if self.channel_first and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        # Denormalize
        if self.normalize != "none":
            image = image * self.std + self.mean

        # Clip and convert to uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        return image

    def batch_transform(
        self,
        images: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform a batch of images.

        Args:
            images: Batch of images [B, H, W, 3]

        Returns:
            Processed batch [B, 3, H, W] or [B, H, W, 3]
        """
        is_tensor = isinstance(images, torch.Tensor)
        if is_tensor:
            device = images.device
            images = images.cpu().numpy()

        processed = np.stack([self.transform(img) for img in images])

        if is_tensor:
            processed = torch.tensor(processed, dtype=torch.float32, device=device)

        return processed
