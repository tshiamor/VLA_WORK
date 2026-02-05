"""
VLM to Action Space Projector

Projects vision-language model embeddings to the action expert's input space.
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class ProjectorConfig:
    """Configuration for the VLM projector."""
    vlm_hidden_size: int = 3584  # Qwen 2.5-VL-7B hidden size
    action_hidden_size: int = 512  # Action expert hidden size
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True


class VLMProjector(nn.Module):
    """
    MLP projector that transforms VLM embeddings to action expert input space.

    Bridges the gap between the high-dimensional VLM representations and
    the action expert's expected input dimension.
    """

    def __init__(self, config: Optional[ProjectorConfig] = None):
        super().__init__()
        self.config = config or ProjectorConfig()

        # Activation function
        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        activation_cls = activation_map.get(self.config.activation, nn.GELU)

        # Build MLP layers
        layers = []
        in_dim = self.config.vlm_hidden_size
        out_dim = self.config.action_hidden_size

        if self.config.num_layers == 1:
            # Single layer projection
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            # Multi-layer MLP with gradual dimension reduction
            hidden_dims = self._compute_hidden_dims(
                in_dim, out_dim, self.config.num_layers
            )

            for i, (dim_in, dim_out) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
                layers.append(nn.Linear(dim_in, dim_out))

                # Add normalization and activation for all but last layer
                if i < len(hidden_dims) - 2:
                    if self.config.use_layer_norm:
                        layers.append(nn.LayerNorm(dim_out))
                    layers.append(activation_cls())
                    layers.append(nn.Dropout(self.config.dropout))

        self.mlp = nn.Sequential(*layers)

        # Final layer norm for stable training
        if self.config.use_layer_norm:
            self.output_norm = nn.LayerNorm(out_dim)
        else:
            self.output_norm = nn.Identity()

        self._init_weights()

    def _compute_hidden_dims(
        self, in_dim: int, out_dim: int, num_layers: int
    ) -> list:
        """Compute intermediate hidden dimensions for gradual reduction."""
        if num_layers <= 1:
            return [in_dim, out_dim]

        # Geometric interpolation between dimensions
        ratio = (out_dim / in_dim) ** (1.0 / num_layers)
        dims = [in_dim]
        for i in range(1, num_layers):
            dims.append(int(in_dim * (ratio ** i)))
        dims.append(out_dim)
        return dims

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        vlm_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project VLM embeddings to action space.

        Args:
            vlm_embeddings: VLM hidden states [B, seq_len, vlm_hidden_size] or [B, vlm_hidden_size]
            attention_mask: Optional attention mask [B, seq_len]

        Returns:
            Projected embeddings [B, seq_len, action_hidden_size] or [B, action_hidden_size]
        """
        # Handle both sequence and pooled inputs
        is_sequence = vlm_embeddings.dim() == 3

        if is_sequence:
            batch_size, seq_len, _ = vlm_embeddings.shape
            # Reshape for MLP: [B * seq_len, hidden_size]
            x = vlm_embeddings.view(-1, self.config.vlm_hidden_size)
            x = self.mlp(x)
            x = self.output_norm(x)
            # Reshape back: [B, seq_len, action_hidden_size]
            x = x.view(batch_size, seq_len, -1)
        else:
            x = self.mlp(vlm_embeddings)
            x = self.output_norm(x)

        return x


class CrossAttentionProjector(nn.Module):
    """
    Alternative projector using cross-attention to selectively attend
    to VLM embeddings for action generation.
    """

    def __init__(
        self,
        vlm_hidden_size: int = 3584,
        action_hidden_size: int = 512,
        num_heads: int = 8,
        num_queries: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vlm_hidden_size = vlm_hidden_size
        self.action_hidden_size = action_hidden_size
        self.num_queries = num_queries

        # Learnable query tokens for action generation
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, action_hidden_size)
        )

        # Project VLM embeddings to action dimension for attention
        self.kv_proj = nn.Linear(vlm_hidden_size, action_hidden_size)

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=action_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(action_hidden_size),
            nn.Linear(action_hidden_size, action_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.query_tokens, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        if self.kv_proj.bias is not None:
            nn.init.zeros_(self.kv_proj.bias)

    def forward(
        self,
        vlm_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project VLM embeddings using cross-attention.

        Args:
            vlm_embeddings: VLM hidden states [B, seq_len, vlm_hidden_size]
            attention_mask: Optional attention mask [B, seq_len]

        Returns:
            Action embeddings [B, num_queries, action_hidden_size]
        """
        batch_size = vlm_embeddings.shape[0]

        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # Project VLM embeddings
        kv = self.kv_proj(vlm_embeddings)

        # Convert attention mask to key padding mask format
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        # Cross-attention
        attended, _ = self.cross_attention(
            query=queries,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
        )

        # Output projection
        output = self.output_proj(attended)

        return output
