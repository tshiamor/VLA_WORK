"""
Data augmentation for VLA training
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import random


@dataclass
class AugmentationConfig:
    """Configuration for VLA data augmentation."""
    # Image augmentations
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: Tuple[float, float] = (-0.1, 0.1)
    noise_std: float = 0.02

    # Geometric augmentations (applied consistently to both cameras)
    random_crop_scale: Tuple[float, float] = (0.9, 1.0)
    random_shift_range: float = 0.05  # Fraction of image size

    # Augmentation probabilities
    color_aug_prob: float = 0.5
    noise_aug_prob: float = 0.3
    crop_aug_prob: float = 0.3

    # Action augmentations
    action_noise_std: float = 0.01
    action_aug_prob: float = 0.2


class VLAAugmentation:
    """
    Data augmentation for VLA training.

    Applies consistent augmentations across multiple camera views
    and appropriate augmentations to actions.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def __call__(
        self,
        context_img: torch.Tensor,
        wrist_img: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentations to a sample.

        Args:
            context_img: Context camera image [C, H, W] or [B, C, H, W]
            wrist_img: Wrist camera image [C, H, W] or [B, C, H, W]
            actions: Optional action tensor [T, action_dim] or [B, T, action_dim]

        Returns:
            Augmented (context_img, wrist_img, actions)
        """
        # Determine if batched
        batched = context_img.dim() == 4

        if batched:
            # Process each sample in batch
            aug_context = []
            aug_wrist = []
            aug_actions = [] if actions is not None else None

            for i in range(context_img.shape[0]):
                ctx, wst, act = self._augment_single(
                    context_img[i],
                    wrist_img[i],
                    actions[i] if actions is not None else None,
                )
                aug_context.append(ctx)
                aug_wrist.append(wst)
                if aug_actions is not None:
                    aug_actions.append(act)

            context_img = torch.stack(aug_context)
            wrist_img = torch.stack(aug_wrist)
            if aug_actions is not None:
                actions = torch.stack(aug_actions)
        else:
            context_img, wrist_img, actions = self._augment_single(
                context_img, wrist_img, actions
            )

        return context_img, wrist_img, actions

    def _augment_single(
        self,
        context_img: torch.Tensor,
        wrist_img: torch.Tensor,
        actions: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Augment a single sample."""

        # Color augmentation (same parameters for both images)
        if random.random() < self.config.color_aug_prob:
            # Sample augmentation parameters once
            brightness = random.uniform(*self.config.brightness_range)
            contrast = random.uniform(*self.config.contrast_range)
            saturation = random.uniform(*self.config.saturation_range)

            context_img = self._color_augment(context_img, brightness, contrast, saturation)
            wrist_img = self._color_augment(wrist_img, brightness, contrast, saturation)

        # Gaussian noise
        if random.random() < self.config.noise_aug_prob:
            noise_std = self.config.noise_std
            context_img = context_img + torch.randn_like(context_img) * noise_std
            wrist_img = wrist_img + torch.randn_like(wrist_img) * noise_std
            context_img = torch.clamp(context_img, 0, 1)
            wrist_img = torch.clamp(wrist_img, 0, 1)

        # Action augmentation
        if actions is not None and random.random() < self.config.action_aug_prob:
            noise = torch.randn_like(actions) * self.config.action_noise_std
            actions = actions + noise

        return context_img, wrist_img, actions

    def _color_augment(
        self,
        img: torch.Tensor,
        brightness: float,
        contrast: float,
        saturation: float,
    ) -> torch.Tensor:
        """Apply color augmentation to an image."""
        # Brightness
        img = img * brightness

        # Contrast
        mean = img.mean()
        img = (img - mean) * contrast + mean

        # Saturation (simplified)
        gray = img.mean(dim=0, keepdim=True)
        img = img * saturation + gray * (1 - saturation)

        return torch.clamp(img, 0, 1)


class TemporalAugmentation:
    """
    Temporal augmentations for action sequences.

    Includes:
    - Random subsampling
    - Speed perturbation
    - Sequence reversal (for symmetric tasks)
    """

    def __init__(
        self,
        subsample_prob: float = 0.2,
        speed_perturb_prob: float = 0.3,
        speed_range: Tuple[float, float] = (0.8, 1.2),
    ):
        self.subsample_prob = subsample_prob
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_range = speed_range

    def __call__(
        self,
        actions: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Apply temporal augmentation to action sequence.

        Args:
            actions: Action sequence [T, action_dim]
            chunk_size: Target output length

        Returns:
            Augmented actions [chunk_size, action_dim]
        """
        T = actions.shape[0]

        # Speed perturbation (resample to different length then crop/pad)
        if random.random() < self.speed_perturb_prob and T >= chunk_size:
            speed = random.uniform(*self.speed_range)
            new_T = int(T * speed)
            new_T = max(new_T, chunk_size)

            # Resample using interpolation
            indices = torch.linspace(0, T - 1, new_T)
            floor_idx = indices.long()
            ceil_idx = torch.clamp(floor_idx + 1, max=T - 1)
            alpha = (indices - floor_idx.float()).unsqueeze(-1)

            actions_resampled = (1 - alpha) * actions[floor_idx] + alpha * actions[ceil_idx]
            actions = actions_resampled

        # Ensure correct length
        if actions.shape[0] > chunk_size:
            # Random crop
            start = random.randint(0, actions.shape[0] - chunk_size)
            actions = actions[start:start + chunk_size]
        elif actions.shape[0] < chunk_size:
            # Pad with last action
            pad_length = chunk_size - actions.shape[0]
            padding = actions[-1:].expand(pad_length, -1)
            actions = torch.cat([actions, padding], dim=0)

        return actions


class InstructionAugmentation:
    """
    Augmentation for language instructions.

    Includes:
    - Synonym replacement
    - Random word dropout
    - Paraphrasing templates
    """

    def __init__(
        self,
        dropout_prob: float = 0.1,
        synonym_prob: float = 0.2,
    ):
        self.dropout_prob = dropout_prob
        self.synonym_prob = synonym_prob

        # Simple synonym dictionary
        self.synonyms = {
            "pick": ["grab", "grasp", "take", "lift"],
            "place": ["put", "set", "position", "drop"],
            "move": ["shift", "transfer", "relocate"],
            "cube": ["block", "box", "object"],
            "red": ["crimson", "scarlet"],
            "green": ["emerald"],
            "goal": ["target", "destination"],
        }

    def __call__(self, instruction: str) -> str:
        """
        Apply augmentation to instruction.

        Args:
            instruction: Original instruction string

        Returns:
            Augmented instruction
        """
        words = instruction.split()
        augmented_words = []

        for word in words:
            # Word dropout
            if random.random() < self.dropout_prob:
                continue

            # Synonym replacement
            word_lower = word.lower().strip('.,!?')
            if word_lower in self.synonyms and random.random() < self.synonym_prob:
                replacement = random.choice(self.synonyms[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                word = replacement

            augmented_words.append(word)

        return ' '.join(augmented_words)
