"""
Action Expert Transformer

Transformer decoder that processes VLM embeddings to produce conditioning
for the flow matching action head. Inspired by Pi0 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class ActionExpertConfig:
    """Configuration for the action expert transformer."""
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_seq_len: int = 256
    # Proprioception
    proprio_dim: int = 14  # joint positions (7) + velocities (7)
    use_proprio: bool = True
    # Action space
    action_dim: int = 7
    chunk_size: int = 16


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, seq_len, hidden_dim]
            key: [B, src_len, hidden_dim] (defaults to query)
            value: [B, src_len, hidden_dim] (defaults to key)
            attention_mask: [B, 1, seq_len, src_len] or broadcastable

        Returns:
            output: [B, seq_len, hidden_dim]
            attention_weights: [B, num_heads, seq_len, src_len]
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len, _ = query.shape
        src_len = key.shape[1]

        # Project to heads
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim)

        # Transpose for attention: [B, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        return output, attention_weights


class TransformerBlock(nn.Module):
    """Single transformer decoder block with self-attention and FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention (for conditioning on VLM)
        self.cross_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input sequence [B, seq_len, hidden_dim]
            context: Context from VLM [B, ctx_len, hidden_dim]
            self_attn_mask: Mask for self-attention
            cross_attn_mask: Mask for cross-attention

        Returns:
            output: [B, seq_len, hidden_dim]
            attention_weights: Dictionary of attention weights
        """
        attention_weights = {}

        # Self-attention with residual
        x_norm = self.self_attn_norm(x)
        self_attn_out, self_attn_weights = self.self_attn(
            x_norm, attention_mask=self_attn_mask
        )
        x = x + self.dropout(self_attn_out)
        attention_weights['self_attn'] = self_attn_weights

        # Cross-attention with residual (if context provided)
        if context is not None:
            x_norm = self.cross_attn_norm(x)
            cross_attn_out, cross_attn_weights = self.cross_attn(
                x_norm, context, context, attention_mask=cross_attn_mask
            )
            x = x + self.dropout(cross_attn_out)
            attention_weights['cross_attn'] = cross_attn_weights

        # FFN with residual
        x_norm = self.ffn_norm(x)
        x = x + self.ffn(x_norm)

        return x, attention_weights


class ActionExpert(nn.Module):
    """
    Transformer-based action expert that processes VLM embeddings
    and produces conditioning for the flow matching head.

    Architecture:
    1. Embeds proprioceptive state (optional)
    2. Processes VLM context through transformer blocks
    3. Outputs conditioning vector for flow matching
    """

    def __init__(self, config: Optional[ActionExpertConfig] = None):
        super().__init__()
        self.config = config or ActionExpertConfig()

        # Proprioception embedding (if used)
        if self.config.use_proprio:
            self.proprio_embed = nn.Sequential(
                nn.Linear(self.config.proprio_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            )

        # Learnable query tokens for action generation
        self.action_queries = nn.Parameter(
            torch.randn(1, self.config.chunk_size, self.config.hidden_dim)
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.config.max_seq_len, self.config.hidden_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                dropout=self.config.dropout,
                attention_dropout=self.config.attention_dropout,
            )
            for _ in range(self.config.num_layers)
        ])

        # Output projection for flow matching conditioning
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.action_queries, mean=0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        vlm_embeddings: torch.Tensor,
        proprio_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process VLM embeddings to produce flow matching conditioning.

        Args:
            vlm_embeddings: Projected VLM embeddings [B, seq_len, hidden_dim]
            proprio_state: Proprioceptive state [B, proprio_dim] (optional)
            attention_mask: Attention mask for VLM embeddings [B, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
            - 'condition': Conditioning for flow matching [B, hidden_dim]
            - 'action_features': Action query features [B, chunk_size, hidden_dim]
            - 'attention_weights': Optional attention weights
        """
        batch_size = vlm_embeddings.shape[0]
        device = vlm_embeddings.device

        # Prepare context sequence
        context_parts = [vlm_embeddings]

        # Add proprioception if available
        if self.config.use_proprio and proprio_state is not None:
            proprio_emb = self.proprio_embed(proprio_state)
            proprio_emb = proprio_emb.unsqueeze(1)  # [B, 1, hidden_dim]
            context_parts.append(proprio_emb)

        # Concatenate context
        context = torch.cat(context_parts, dim=1)

        # Add positional embeddings to context
        context_len = context.shape[1]
        context = context + self.pos_embed[:, :context_len, :]

        # Expand action queries for batch
        action_queries = self.action_queries.expand(batch_size, -1, -1)

        # Add positional embeddings to queries
        query_pos = self.pos_embed[:, :self.config.chunk_size, :]
        x = action_queries + query_pos

        # Process through transformer blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(
                x,
                context=context,
                cross_attn_mask=None,  # Can add mask if needed
            )
            if return_attention:
                all_attention_weights.append(attn_weights)

        # Project to conditioning space
        action_features = self.output_proj(x)  # [B, chunk_size, hidden_dim]

        # Pool action features for flow matching conditioning
        condition = action_features.mean(dim=1)  # [B, hidden_dim]

        result = {
            'condition': condition,
            'action_features': action_features,
        }

        if return_attention:
            result['attention_weights'] = all_attention_weights

        return result


class LightweightActionExpert(nn.Module):
    """
    Lightweight action expert using only MLPs and attention pooling.
    Faster alternative to full transformer for resource-constrained settings.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        proprio_dim: int = 14,
        use_proprio: bool = True,
        num_layers: int = 3,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.use_proprio = use_proprio

        # Proprioception embedding
        if use_proprio:
            self.proprio_embed = nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        # Attention pooling for VLM embeddings
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        # MLP layers
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ])
        self.mlp = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim * chunk_size)

    def forward(
        self,
        vlm_embeddings: torch.Tensor,
        proprio_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vlm_embeddings: [B, seq_len, hidden_dim]
            proprio_state: [B, proprio_dim]
            attention_mask: [B, seq_len]

        Returns:
            Dictionary with 'condition' and 'action_features'
        """
        batch_size = vlm_embeddings.shape[0]

        # Attention pooling over VLM sequence
        attn_weights = self.attention_pool(vlm_embeddings)  # [B, seq_len, 1]
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), float('-inf')
            )
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (vlm_embeddings * attn_weights).sum(dim=1)  # [B, hidden_dim]

        # Add proprioception
        if self.use_proprio and proprio_state is not None:
            proprio_emb = self.proprio_embed(proprio_state)
            pooled = pooled + proprio_emb

        # Process through MLP
        features = self.mlp(pooled)

        # Project to action features
        action_features = self.output_proj(features)
        action_features = action_features.view(batch_size, self.chunk_size, -1)

        condition = features  # Use MLP output as condition

        return {
            'condition': condition,
            'action_features': action_features,
        }
