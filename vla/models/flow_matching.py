"""
Flow Matching for Action Generation

Implements Conditional Flow Matching (CFM) for continuous action generation.
Based on "Flow Matching for Generative Modeling" (Lipman et al., 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching."""
    action_dim: int = 7  # 7-DOF for Franka/UR5e
    hidden_dim: int = 512
    num_steps: int = 100  # Number of flow steps for inference
    sigma_min: float = 1e-4  # Minimum noise level
    sigma_max: float = 1.0  # Maximum noise level
    ode_solver: str = "euler"  # euler, heun, or rk4
    # Action chunking
    chunk_size: int = 16  # Predict 16 timesteps at once


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [B] in range [0, 1]

        Returns:
            Positional embeddings [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class FlowMatchingActionHead(nn.Module):
    """
    Flow Matching head for action generation.

    Uses Conditional Flow Matching to learn a vector field that transforms
    noise to action distributions conditioned on VLM embeddings.
    """

    def __init__(self, config: Optional[FlowMatchingConfig] = None):
        super().__init__()
        self.config = config or FlowMatchingConfig()

        # Total action dimension (chunk_size * action_dim)
        self.total_action_dim = self.config.chunk_size * self.config.action_dim

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim * 4, self.config.hidden_dim),
        )

        # Conditioning projection (from action expert)
        self.cond_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)

        # Action projection (noisy action input)
        self.action_proj = nn.Linear(self.total_action_dim, self.config.hidden_dim)

        # Vector field network (predicts velocity)
        self.velocity_net = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim * 2),
            nn.LayerNorm(self.config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim * 2),
            nn.LayerNorm(self.config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.total_action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the velocity field at time t.

        Args:
            x_t: Noisy action at time t [B, total_action_dim]
            t: Time values [B] in range [0, 1]
            condition: Conditioning embedding from action expert [B, hidden_dim]

        Returns:
            Predicted velocity [B, total_action_dim]
        """
        # Embed time
        t_emb = self.time_emb(t)

        # Project condition and action
        cond_emb = self.cond_proj(condition)
        action_emb = self.action_proj(x_t)

        # Concatenate all embeddings
        combined = torch.cat([t_emb, cond_emb, action_emb], dim=-1)

        # Predict velocity
        velocity = self.velocity_net(combined)

        return velocity

    def compute_loss(
        self,
        x_1: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute flow matching loss.

        Uses optimal transport conditional flow matching:
        - Sample t uniformly from [0, 1]
        - Sample x_0 from prior (standard Gaussian)
        - Interpolate x_t = (1 - t) * x_0 + t * x_1
        - Target velocity: v_t = x_1 - x_0
        - Loss: ||v_theta(x_t, t, c) - v_t||^2

        Args:
            x_1: Target actions [B, chunk_size, action_dim]
            condition: Conditioning embedding [B, hidden_dim]

        Returns:
            loss: Scalar flow matching loss
            metrics: Dictionary of additional metrics
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # Flatten action chunks
        x_1_flat = x_1.view(batch_size, -1)  # [B, total_action_dim]

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Sample from prior (standard Gaussian)
        x_0 = torch.randn_like(x_1_flat)

        # Interpolate
        t_expand = t[:, None]
        x_t = (1 - t_expand) * x_0 + t_expand * x_1_flat

        # Target velocity (optimal transport)
        v_target = x_1_flat - x_0

        # Predicted velocity
        v_pred = self.forward(x_t, t, condition)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        # Additional metrics
        metrics = {
            "mse": loss.detach(),
            "v_pred_norm": v_pred.norm(dim=-1).mean().detach(),
            "v_target_norm": v_target.norm(dim=-1).mean().detach(),
        }

        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample actions using ODE integration.

        Args:
            condition: Conditioning embedding [B, hidden_dim]
            num_steps: Number of integration steps (overrides config)

        Returns:
            Sampled actions [B, chunk_size, action_dim]
        """
        num_steps = num_steps or self.config.num_steps
        batch_size = condition.shape[0]
        device = condition.device

        # Start from prior
        x_t = torch.randn(batch_size, self.total_action_dim, device=device)

        # Time steps
        dt = 1.0 / num_steps

        # ODE integration
        if self.config.ode_solver == "euler":
            for i in range(num_steps):
                t = torch.full((batch_size,), i / num_steps, device=device)
                v = self.forward(x_t, t, condition)
                x_t = x_t + v * dt

        elif self.config.ode_solver == "heun":
            for i in range(num_steps):
                t = torch.full((batch_size,), i / num_steps, device=device)
                t_next = torch.full((batch_size,), (i + 1) / num_steps, device=device)

                # Euler step
                v1 = self.forward(x_t, t, condition)
                x_euler = x_t + v1 * dt

                # Correction
                v2 = self.forward(x_euler, t_next, condition)
                x_t = x_t + 0.5 * (v1 + v2) * dt

        elif self.config.ode_solver == "rk4":
            for i in range(num_steps):
                t = i / num_steps
                t_mid = (i + 0.5) / num_steps
                t_next = (i + 1) / num_steps

                t_tensor = torch.full((batch_size,), t, device=device)
                t_mid_tensor = torch.full((batch_size,), t_mid, device=device)
                t_next_tensor = torch.full((batch_size,), t_next, device=device)

                k1 = self.forward(x_t, t_tensor, condition)
                k2 = self.forward(x_t + 0.5 * dt * k1, t_mid_tensor, condition)
                k3 = self.forward(x_t + 0.5 * dt * k2, t_mid_tensor, condition)
                k4 = self.forward(x_t + dt * k3, t_next_tensor, condition)

                x_t = x_t + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Reshape to action chunks
        actions = x_t.view(batch_size, self.config.chunk_size, self.config.action_dim)

        return actions


class ConditionalFlowMatcher:
    """
    Utility class for conditional flow matching operations.

    Provides methods for:
    - Computing optimal transport paths
    - Training objective computation
    - Various flow matching formulations
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        distribution: str = "uniform",
    ) -> torch.Tensor:
        """
        Sample time values for training.

        Args:
            batch_size: Number of samples
            device: Target device
            distribution: "uniform", "logit_normal", or "beta"

        Returns:
            Time values [batch_size] in range [0, 1]
        """
        if distribution == "uniform":
            t = torch.rand(batch_size, device=device)
        elif distribution == "logit_normal":
            # Logit-normal distribution (more weight on t near 0 and 1)
            u = torch.randn(batch_size, device=device) * 0.5
            t = torch.sigmoid(u)
        elif distribution == "beta":
            # Beta distribution with mode at 0.5
            alpha = torch.tensor([2.0])
            beta = torch.tensor([2.0])
            dist = torch.distributions.Beta(alpha, beta)
            t = dist.sample((batch_size,)).squeeze(-1).to(device)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return t

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolate between x_0 and x_1 at time t.

        Args:
            x_0: Source samples (noise) [B, ...]
            x_1: Target samples (data) [B, ...]
            t: Time values [B]

        Returns:
            Interpolated samples [B, ...]
        """
        # Expand t for broadcasting
        while t.dim() < x_0.dim():
            t = t.unsqueeze(-1)

        return (1 - t) * x_0 + t * x_1

    def compute_target_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target velocity for optimal transport path.

        For OT-CFM: v(x, t) = x_1 - x_0 (constant velocity)

        Args:
            x_0: Source samples
            x_1: Target samples

        Returns:
            Target velocity
        """
        return x_1 - x_0

    def add_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        sigma: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Add noise to interpolated samples (for stochastic CFM).

        Args:
            x_t: Interpolated samples
            t: Time values
            sigma: Noise scale (uses config if not provided)

        Returns:
            Noisy samples
        """
        if sigma is None:
            # Schedule noise based on time
            while t.dim() < x_t.dim():
                t = t.unsqueeze(-1)
            sigma = self.sigma_min + t * (self.sigma_max - self.sigma_min)

        noise = torch.randn_like(x_t)
        return x_t + sigma * noise
