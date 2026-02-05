"""
Vision-Language-Action (VLA) Model

Combines Qwen 2.5-VL vision-language model with a flow matching
action expert for robot manipulation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

from .vision_encoder import QwenVisionEncoder, VisionEncoderConfig
from .projector import VLMProjector, ProjectorConfig, CrossAttentionProjector
from .action_expert import ActionExpert, ActionExpertConfig, LightweightActionExpert
from .flow_matching import FlowMatchingActionHead, FlowMatchingConfig


@dataclass
class VLAConfig:
    """Configuration for the complete VLA model."""
    # Vision encoder config
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    # Projector config
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    # Action expert config
    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    # Flow matching config
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    # Model options
    use_cross_attention_projector: bool = False
    use_lightweight_action_expert: bool = False
    # Action space
    action_dim: int = 7  # 7-DOF
    chunk_size: int = 16
    # Normalization
    action_mean: Optional[List[float]] = None
    action_std: Optional[List[float]] = None

    def __post_init__(self):
        # Ensure consistency across configs
        self.action_expert.action_dim = self.action_dim
        self.action_expert.chunk_size = self.chunk_size
        self.flow_matching.action_dim = self.action_dim
        self.flow_matching.chunk_size = self.chunk_size
        self.projector.action_hidden_size = self.action_expert.hidden_dim
        self.flow_matching.hidden_dim = self.action_expert.hidden_dim


class VLAModel(nn.Module):
    """
    Vision-Language-Action Model

    Architecture:
    1. Qwen 2.5-VL processes images + language instruction
    2. Projector maps VLM embeddings to action space
    3. Action expert (transformer) produces conditioning
    4. Flow matching head generates action chunks

    Training strategy:
    - Frozen VLM backbone with LoRA adapters
    - Trainable projector, action expert, and flow head
    """

    def __init__(self, config: Optional[VLAConfig] = None):
        super().__init__()
        self.config = config or VLAConfig()

        # Vision-language encoder
        self.vision_encoder = QwenVisionEncoder(self.config.vision_encoder)

        # Projector: VLM embeddings -> action space
        if self.config.use_cross_attention_projector:
            self.projector = CrossAttentionProjector(
                vlm_hidden_size=self.config.projector.vlm_hidden_size,
                action_hidden_size=self.config.projector.action_hidden_size,
            )
        else:
            self.projector = VLMProjector(self.config.projector)

        # Action expert
        if self.config.use_lightweight_action_expert:
            self.action_expert = LightweightActionExpert(
                hidden_dim=self.config.action_expert.hidden_dim,
                proprio_dim=self.config.action_expert.proprio_dim,
                use_proprio=self.config.action_expert.use_proprio,
                chunk_size=self.config.chunk_size,
            )
        else:
            self.action_expert = ActionExpert(self.config.action_expert)

        # Flow matching action head
        self.flow_head = FlowMatchingActionHead(self.config.flow_matching)

        # Action normalization
        self.register_buffer(
            'action_mean',
            torch.zeros(self.config.action_dim)
            if self.config.action_mean is None
            else torch.tensor(self.config.action_mean)
        )
        self.register_buffer(
            'action_std',
            torch.ones(self.config.action_dim)
            if self.config.action_std is None
            else torch.tensor(self.config.action_std)
        )

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to zero mean and unit variance."""
        return (actions - self.action_mean) / (self.action_std + 1e-8)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions back to original scale."""
        return actions * self.action_std + self.action_mean

    def encode_observation(
        self,
        images: List[Any],
        instruction: str,
        proprio_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode visual observations and instruction.

        Args:
            images: List of images [context_cam, wrist_cam]
            instruction: Language instruction
            proprio_state: Robot proprioceptive state [B, proprio_dim]

        Returns:
            Dictionary with encoded features
        """
        # Get VLM embeddings
        vlm_outputs = self.vision_encoder(
            images=images,
            instruction=instruction,
            return_last_hidden_state=True,
        )

        # Project to action space
        vlm_embeddings = vlm_outputs.get('last_hidden_state', vlm_outputs.get('pooled_output'))
        if vlm_embeddings is None:
            raise ValueError("VLM did not return embeddings")

        projected = self.projector(vlm_embeddings)

        # Get action conditioning
        expert_outputs = self.action_expert(
            vlm_embeddings=projected,
            proprio_state=proprio_state,
        )

        return {
            'vlm_embeddings': vlm_embeddings,
            'projected': projected,
            'condition': expert_outputs['condition'],
            'action_features': expert_outputs['action_features'],
        }

    def forward(
        self,
        images: Optional[List[Any]] = None,
        instruction: Optional[str] = None,
        proprio_state: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
        vlm_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VLA model.

        For training: provide target_actions to compute loss
        For inference: omit target_actions to sample actions

        Args:
            images: List of images [context_cam, wrist_cam]
            instruction: Language instruction
            proprio_state: Robot proprioceptive state [B, proprio_dim]
            target_actions: Target action chunks [B, chunk_size, action_dim]
            vlm_inputs: Pre-processed VLM inputs (alternative to images+instruction)

        Returns:
            Dictionary containing loss (training) or sampled actions (inference)
        """
        # Get VLM embeddings
        if vlm_inputs is not None:
            vlm_outputs = self.vision_encoder(
                inputs=vlm_inputs,
                return_last_hidden_state=True,
            )
        elif images is not None and instruction is not None:
            vlm_outputs = self.vision_encoder(
                images=images,
                instruction=instruction,
                return_last_hidden_state=True,
            )
        else:
            raise ValueError("Either 'vlm_inputs' or both 'images' and 'instruction' required")

        # Project VLM embeddings
        vlm_embeddings = vlm_outputs.get('last_hidden_state', vlm_outputs.get('pooled_output'))
        projected = self.projector(vlm_embeddings)

        # Get action conditioning
        expert_outputs = self.action_expert(
            vlm_embeddings=projected,
            proprio_state=proprio_state,
        )
        condition = expert_outputs['condition']

        # Training: compute flow matching loss
        if target_actions is not None:
            # Normalize target actions
            normalized_actions = self.normalize_actions(target_actions)

            # Compute flow matching loss
            loss, metrics = self.flow_head.compute_loss(
                x_1=normalized_actions,
                condition=condition,
            )

            return {
                'loss': loss,
                'metrics': metrics,
            }

        # Inference: sample actions
        else:
            normalized_actions = self.flow_head.sample(condition)
            actions = self.denormalize_actions(normalized_actions)

            return {
                'actions': actions,  # [B, chunk_size, action_dim]
                'condition': condition,
            }

    @torch.no_grad()
    def predict_action(
        self,
        images: List[Any],
        instruction: str,
        proprio_state: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Predict actions for deployment.

        Args:
            images: List of images
            instruction: Language instruction
            proprio_state: Current robot state
            num_samples: Number of action samples to generate

        Returns:
            Actions [num_samples, chunk_size, action_dim]
        """
        self.eval()

        # Encode observation (handles single sample)
        encoded = self.encode_observation(images, instruction, proprio_state)
        condition = encoded['condition']

        # Expand condition for multiple samples
        if num_samples > 1:
            condition = condition.expand(num_samples, -1)

        # Sample actions
        normalized_actions = self.flow_head.sample(condition)
        actions = self.denormalize_actions(normalized_actions)

        return actions

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = []

        # VLM LoRA parameters
        params.extend(self.vision_encoder.get_trainable_parameters())

        # Projector parameters (always trainable)
        params.extend(self.projector.parameters())

        # Action expert parameters (always trainable)
        params.extend(self.action_expert.parameters())

        # Flow head parameters (always trainable)
        params.extend(self.flow_head.parameters())

        return params

    def save_checkpoint(self, path: str, include_vlm: bool = False):
        """
        Save model checkpoint.

        Args:
            path: Save path
            include_vlm: Whether to include full VLM weights (large!)
        """
        checkpoint = {
            'config': self.config,
            'projector': self.projector.state_dict(),
            'action_expert': self.action_expert.state_dict(),
            'flow_head': self.flow_head.state_dict(),
            'action_mean': self.action_mean,
            'action_std': self.action_std,
        }

        # Save LoRA weights
        if self.config.vision_encoder.use_lora:
            checkpoint['lora_weights'] = self.vision_encoder.lora_modules.state_dict()

        # Optionally include full VLM
        if include_vlm:
            checkpoint['vlm_state_dict'] = self.vision_encoder.model.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_vlm: bool = False):
        """
        Load model checkpoint.

        Args:
            path: Checkpoint path
            load_vlm: Whether to load full VLM weights
        """
        checkpoint = torch.load(path, map_location='cpu')

        # Load component weights
        self.projector.load_state_dict(checkpoint['projector'])
        self.action_expert.load_state_dict(checkpoint['action_expert'])
        self.flow_head.load_state_dict(checkpoint['flow_head'])

        # Load normalization stats
        if 'action_mean' in checkpoint:
            self.action_mean = checkpoint['action_mean']
        if 'action_std' in checkpoint:
            self.action_std = checkpoint['action_std']

        # Load LoRA weights
        if 'lora_weights' in checkpoint:
            self.vision_encoder.initialize()  # Ensure model is loaded
            self.vision_encoder.lora_modules.load_state_dict(checkpoint['lora_weights'])

        # Optionally load full VLM
        if load_vlm and 'vlm_state_dict' in checkpoint:
            self.vision_encoder.initialize()
            self.vision_encoder.model.load_state_dict(checkpoint['vlm_state_dict'])


def create_vla_model(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    action_dim: int = 7,
    chunk_size: int = 16,
    hidden_dim: int = 512,
    freeze_vlm: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    device_map: str = "auto",
) -> VLAModel:
    """
    Factory function to create a VLA model with common configurations.

    Args:
        model_name: Qwen VL model name
        action_dim: Robot action dimension
        chunk_size: Number of action steps to predict
        hidden_dim: Hidden dimension for action expert
        freeze_vlm: Whether to freeze VLM backbone
        use_lora: Whether to use LoRA adapters
        lora_r: LoRA rank
        device_map: Device mapping for VLM

    Returns:
        Configured VLAModel
    """
    config = VLAConfig(
        vision_encoder=VisionEncoderConfig(
            model_name=model_name,
            freeze_backbone=freeze_vlm,
            use_lora=use_lora,
            lora_r=lora_r,
            device_map=device_map,
        ),
        projector=ProjectorConfig(
            vlm_hidden_size=3584,  # Qwen 2.5-VL-7B
            action_hidden_size=hidden_dim,
        ),
        action_expert=ActionExpertConfig(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
        ),
        flow_matching=FlowMatchingConfig(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            chunk_size=chunk_size,
        ),
        action_dim=action_dim,
        chunk_size=chunk_size,
    )

    return VLAModel(config)
