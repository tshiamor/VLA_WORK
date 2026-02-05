"""
Loss functions for VLA training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching loss for action generation.

    Computes the MSE between predicted and target velocity fields
    in the optimal transport formulation.
    """

    def __init__(
        self,
        reduction: str = "mean",
        velocity_weight: float = 1.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.velocity_weight = velocity_weight

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            v_pred: Predicted velocity [B, action_dim] or [B, chunk_size, action_dim]
            v_target: Target velocity (same shape)
            weights: Optional per-sample weights [B]

        Returns:
            Loss value
        """
        # MSE loss per sample
        loss = F.mse_loss(v_pred, v_target, reduction='none')

        # Sum over action dimensions
        if loss.dim() == 3:
            loss = loss.sum(dim=[1, 2])  # [B]
        else:
            loss = loss.sum(dim=1)  # [B]

        # Apply weights if provided
        if weights is not None:
            loss = loss * weights

        # Reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss * self.velocity_weight


class VLALoss(nn.Module):
    """
    Combined loss for VLA training.

    Includes:
    - Flow matching loss for action prediction
    - Optional auxiliary losses (action smoothness, etc.)
    """

    def __init__(
        self,
        flow_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        chunk_consistency_weight: float = 0.05,
    ):
        super().__init__()
        self.flow_weight = flow_weight
        self.smoothness_weight = smoothness_weight
        self.chunk_consistency_weight = chunk_consistency_weight

        self.flow_loss = FlowMatchingLoss()

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        actions_pred: Optional[torch.Tensor] = None,
        actions_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined VLA loss.

        Args:
            v_pred: Predicted velocity for flow matching
            v_target: Target velocity
            actions_pred: Predicted action chunk [B, chunk_size, action_dim]
            actions_target: Target action chunk (same shape)

        Returns:
            total_loss: Combined loss
            metrics: Dictionary of individual loss components
        """
        metrics = {}

        # Flow matching loss
        flow_loss = self.flow_loss(v_pred, v_target)
        metrics['flow_loss'] = flow_loss.detach()

        total_loss = self.flow_weight * flow_loss

        # Action smoothness loss (if predictions available)
        if actions_pred is not None and self.smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness_loss(actions_pred)
            metrics['smoothness_loss'] = smoothness_loss.detach()
            total_loss = total_loss + self.smoothness_weight * smoothness_loss

        # Chunk consistency loss
        if actions_pred is not None and actions_target is not None:
            if self.chunk_consistency_weight > 0:
                consistency_loss = F.mse_loss(actions_pred, actions_target)
                metrics['consistency_loss'] = consistency_loss.detach()
                total_loss = total_loss + self.chunk_consistency_weight * consistency_loss

        metrics['total_loss'] = total_loss.detach()

        return total_loss, metrics

    def _compute_smoothness_loss(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute action smoothness loss (penalize jerkiness).

        Args:
            actions: Action sequence [B, T, action_dim]

        Returns:
            Smoothness loss value
        """
        if actions.shape[1] < 2:
            return torch.tensor(0.0, device=actions.device)

        # First derivative (velocity)
        velocity = actions[:, 1:] - actions[:, :-1]

        # Second derivative (acceleration)
        if actions.shape[1] < 3:
            return (velocity ** 2).mean()

        acceleration = velocity[:, 1:] - velocity[:, :-1]

        # Penalize large accelerations
        smoothness_loss = (acceleration ** 2).mean()

        return smoothness_loss


class ActionChunkLoss(nn.Module):
    """
    Loss for action chunk prediction with temporal weighting.

    Applies higher weight to earlier timesteps in the chunk
    since they're more relevant for immediate execution.
    """

    def __init__(
        self,
        chunk_size: int = 16,
        temporal_decay: float = 0.95,
        reduction: str = "mean",
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.temporal_decay = temporal_decay
        self.reduction = reduction

        # Pre-compute temporal weights
        weights = torch.tensor([temporal_decay ** i for i in range(chunk_size)])
        weights = weights / weights.sum() * chunk_size  # Normalize
        self.register_buffer('temporal_weights', weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporally-weighted action chunk loss.

        Args:
            pred: Predicted actions [B, chunk_size, action_dim]
            target: Target actions [B, chunk_size, action_dim]

        Returns:
            Loss value
        """
        # Per-timestep MSE
        mse = F.mse_loss(pred, target, reduction='none')  # [B, T, action_dim]
        mse = mse.mean(dim=-1)  # [B, T]

        # Apply temporal weights
        weighted_mse = mse * self.temporal_weights.unsqueeze(0)

        if self.reduction == "mean":
            return weighted_mse.mean()
        elif self.reduction == "sum":
            return weighted_mse.sum()
        else:
            return weighted_mse


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative embeddings.

    Can be used for instruction-action alignment.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        Args:
            embeddings1: First set of embeddings [B, D]
            embeddings2: Second set of embeddings [B, D]
            labels: Optional positive pair indicators

        Returns:
            Loss value
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=-1)
        embeddings2 = F.normalize(embeddings2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        # Labels: diagonal elements are positives
        if labels is None:
            labels = torch.arange(similarity.shape[0], device=similarity.device)

        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)

        return loss
