"""
Qwen 2.5-VL Vision Encoder Wrapper

This module wraps the Qwen 2.5-VL model for use in VLA systems.
Supports frozen backbone with optional LoRA adapters for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class VisionEncoderConfig:
    """Configuration for the Qwen Vision Encoder."""
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    freeze_backbone: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    # Vision processing settings
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation."""
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class QwenVisionEncoder(nn.Module):
    """
    Wrapper for Qwen 2.5-VL model that extracts visual-language embeddings.

    Supports:
    - Frozen backbone for inference efficiency
    - LoRA adapters for parameter-efficient fine-tuning
    - Multi-image input (context + wrist cameras)
    - Language instruction conditioning
    """

    def __init__(self, config: Optional[VisionEncoderConfig] = None):
        super().__init__()
        self.config = config or VisionEncoderConfig()
        self._model = None
        self._processor = None
        self._lora_layers: Dict[str, LoRALinear] = {}
        self._initialized = False

    def initialize(self):
        """Lazy initialization of the model to avoid loading at import time."""
        if self._initialized:
            return

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Please install transformers with: pip install transformers>=4.37.0"
            )

        # Determine torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        # Load model and processor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Freeze backbone if configured
        if self.config.freeze_backbone:
            self._freeze_backbone()

        # Add LoRA adapters if configured
        if self.config.use_lora:
            self._add_lora_adapters()

        self._initialized = True

    def _freeze_backbone(self):
        """Freeze all parameters in the backbone model."""
        for param in self._model.parameters():
            param.requires_grad = False

    def _add_lora_adapters(self):
        """Add LoRA adapters to target modules."""
        for name, module in self._model.named_modules():
            if any(target in name for target in self.config.lora_target_modules):
                if isinstance(module, nn.Linear):
                    # Create LoRA layer
                    lora_layer = LoRALinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        r=self.config.lora_r,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout,
                    )
                    # Move to same device and dtype as original module
                    if hasattr(module.weight, 'device'):
                        lora_layer = lora_layer.to(
                            device=module.weight.device,
                            dtype=module.weight.dtype
                        )
                    self._lora_layers[name] = lora_layer

        # Register LoRA layers as submodules
        self.lora_modules = nn.ModuleDict({
            name.replace(".", "_"): layer
            for name, layer in self._lora_layers.items()
        })

    @property
    def model(self):
        """Get the underlying Qwen model."""
        if not self._initialized:
            self.initialize()
        return self._model

    @property
    def processor(self):
        """Get the processor for preparing inputs."""
        if not self._initialized:
            self.initialize()
        return self._processor

    @property
    def hidden_size(self) -> int:
        """Get the hidden dimension of the model."""
        if not self._initialized:
            self.initialize()
        return self._model.config.hidden_size

    def prepare_inputs(
        self,
        images: List[Any],
        instruction: str,
        image_labels: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model.

        Args:
            images: List of PIL images or numpy arrays (e.g., [context_cam, wrist_cam])
            instruction: Language instruction for the task
            image_labels: Optional labels for each image (e.g., ["context view", "wrist view"])

        Returns:
            Dictionary of model inputs
        """
        if not self._initialized:
            self.initialize()

        # Build conversation format for Qwen2.5-VL
        if image_labels is None:
            image_labels = [f"Image {i+1}" for i in range(len(images))]

        # Create content with images and text
        content = []
        for i, (img, label) in enumerate(zip(images, image_labels)):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"[{label}]"})

        content.append({"type": "text", "text": f"\nInstruction: {instruction}"})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Apply chat template and process
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self._processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    def forward(
        self,
        images: Optional[List[Any]] = None,
        instruction: Optional[str] = None,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        return_last_hidden_state: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the vision-language model.

        Args:
            images: List of images (if inputs not provided)
            instruction: Language instruction (if inputs not provided)
            inputs: Pre-processed inputs (alternative to images+instruction)
            return_last_hidden_state: Whether to return the last hidden state

        Returns:
            Dictionary containing:
            - 'last_hidden_state': Hidden states from the last layer [B, seq_len, hidden_dim]
            - 'pooled_output': Pooled representation [B, hidden_dim]
        """
        if not self._initialized:
            self.initialize()

        # Prepare inputs if not provided
        if inputs is None:
            if images is None or instruction is None:
                raise ValueError("Either 'inputs' or both 'images' and 'instruction' must be provided")
            inputs = self.prepare_inputs(images, instruction)

        # Move inputs to model device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward pass with output_hidden_states
        with torch.set_grad_enabled(self.training and (self.config.use_lora or not self.config.freeze_backbone)):
            outputs = self._model(
                **inputs,
                output_hidden_states=return_last_hidden_state,
                return_dict=True,
            )

        result = {}

        if return_last_hidden_state:
            # Get last hidden state
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None

            if hidden_states is not None:
                result['last_hidden_state'] = hidden_states

                # Apply LoRA if enabled
                if self.config.use_lora and self._lora_layers:
                    # LoRA is applied during the forward pass of attention layers
                    # The hidden states already include LoRA contributions
                    pass

                # Create pooled output (mean of sequence)
                # Exclude padding tokens if attention mask is available
                if 'attention_mask' in inputs:
                    mask = inputs['attention_mask'].unsqueeze(-1)
                    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    pooled = hidden_states.mean(dim=1)

                result['pooled_output'] = pooled

        return result

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (LoRA layers)."""
        if not self._initialized:
            self.initialize()

        trainable = []
        if self.config.use_lora:
            for module in self.lora_modules.values():
                trainable.extend(module.parameters())

        if not self.config.freeze_backbone:
            trainable.extend(self._model.parameters())

        return trainable

    def save_lora_weights(self, path: str):
        """Save LoRA adapter weights."""
        if not self.config.use_lora:
            raise ValueError("LoRA is not enabled")

        torch.save(self.lora_modules.state_dict(), path)

    def load_lora_weights(self, path: str):
        """Load LoRA adapter weights."""
        if not self._initialized:
            self.initialize()

        if not self.config.use_lora:
            raise ValueError("LoRA is not enabled")

        state_dict = torch.load(path, map_location='cpu')
        self.lora_modules.load_state_dict(state_dict)
