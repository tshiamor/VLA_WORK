"""
Qwen 2.5-VL Vision Encoder Wrapper

This module wraps the Qwen 2.5-VL model for use in VLA systems.
Supports frozen backbone with optional LoRA adapters for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np


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
    # For batched training
    use_4bit: bool = False  # Enable 4-bit quantization for memory efficiency


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
    - Batched training with proper image processing
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

        # Load model with optional quantization
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }

        if self.config.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("Warning: bitsandbytes not available, skipping 4-bit quantization")

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            **model_kwargs,
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
        """Add LoRA adapters to target modules and hook them into the forward pass."""
        self._lora_hooks = []

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

                    # Register forward hook to apply LoRA during forward pass
                    def _make_lora_hook(lora):
                        def hook(module, input, output):
                            return output + lora(input[0])
                        return hook

                    handle = module.register_forward_hook(_make_lora_hook(lora_layer))
                    self._lora_hooks.append(handle)

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

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor image to PIL Image."""
        # Handle different tensor formats
        if tensor.dim() == 3:
            # [C, H, W] or [H, W, C]
            if tensor.shape[0] in [1, 3, 4]:  # Channel first
                tensor = tensor.permute(1, 2, 0)
            # Now [H, W, C]
            tensor = tensor.cpu()
            if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
                tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
            return Image.fromarray(tensor.numpy())
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    def prepare_inputs_batch(
        self,
        context_images: torch.Tensor,  # [B, C, H, W]
        wrist_images: torch.Tensor,    # [B, C, H, W]
        instructions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batched inputs for the model.

        This processes each sample individually through the Qwen processor
        then stacks them for batched forward pass.
        """
        if not self._initialized:
            self.initialize()

        batch_size = context_images.shape[0]
        all_inputs = []

        for i in range(batch_size):
            # Convert tensors to PIL images
            context_pil = self._tensor_to_pil(context_images[i])
            wrist_pil = self._tensor_to_pil(wrist_images[i])
            instruction = instructions[i] if isinstance(instructions, list) else instructions

            # Prepare single sample
            inputs = self.prepare_inputs(
                images=[context_pil, wrist_pil],
                instruction=instruction,
            )
            all_inputs.append(inputs)

        # Stack inputs (this is tricky because different samples may have different sequence lengths)
        # For now, we'll process samples one at a time in forward
        return {'batch_inputs': all_inputs}

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

        # Convert numpy arrays to PIL if needed
        processed_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                processed_images.append(Image.fromarray(img))
            elif isinstance(img, torch.Tensor):
                processed_images.append(self._tensor_to_pil(img))
            else:
                processed_images.append(img)

        # Build conversation format for Qwen2.5-VL
        if image_labels is None:
            image_labels = ["Context view", "Wrist view"]

        # Create content with images and text
        content = []
        for i, (img, label) in enumerate(zip(processed_images, image_labels)):
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
            images=processed_images,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    def forward(
        self,
        images: Optional[List[Any]] = None,
        instruction: Optional[str] = None,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        # Batched inputs
        context_images: Optional[torch.Tensor] = None,
        wrist_images: Optional[torch.Tensor] = None,
        instructions: Optional[List[str]] = None,
        return_last_hidden_state: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the vision-language model.

        Supports both single-sample and batched inputs.

        For batched training:
            - context_images: [B, C, H, W]
            - wrist_images: [B, C, H, W]
            - instructions: List of B strings

        For single sample:
            - images: [context_pil, wrist_pil]
            - instruction: str
        """
        if not self._initialized:
            self.initialize()

        device = next(self._model.parameters()).device

        # Handle batched inputs for training
        if context_images is not None and wrist_images is not None:
            return self._forward_batch(
                context_images, wrist_images, instructions, device, return_last_hidden_state
            )

        # Handle single sample or pre-processed inputs
        if inputs is None:
            if images is None or instruction is None:
                raise ValueError("Either 'inputs' or both 'images' and 'instruction' must be provided")
            inputs = self.prepare_inputs(images, instruction)

        # Move inputs to model device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward pass with output_hidden_states
        with torch.set_grad_enabled(self.training and (self.config.use_lora or not self.config.freeze_backbone)):
            outputs = self._model(
                **inputs,
                output_hidden_states=return_last_hidden_state,
                return_dict=True,
            )

        return self._extract_outputs(outputs, inputs, return_last_hidden_state)

    def _forward_batch(
        self,
        context_images: torch.Tensor,
        wrist_images: torch.Tensor,
        instructions: List[str],
        device: torch.device,
        return_last_hidden_state: bool,
    ) -> Dict[str, torch.Tensor]:
        """Process a batch of images by iterating through samples."""
        batch_size = context_images.shape[0]
        all_hidden_states = []
        all_pooled = []

        for i in range(batch_size):
            # Convert single sample tensors to PIL
            context_pil = self._tensor_to_pil(context_images[i])
            wrist_pil = self._tensor_to_pil(wrist_images[i])

            # Get instruction for this sample
            instruction = instructions[i] if isinstance(instructions, list) else instructions

            # Prepare inputs for single sample
            inputs = self.prepare_inputs(
                images=[context_pil, wrist_pil],
                instruction=instruction,
            )

            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Forward pass
            with torch.set_grad_enabled(self.training and (self.config.use_lora or not self.config.freeze_backbone)):
                outputs = self._model(
                    **inputs,
                    output_hidden_states=return_last_hidden_state,
                    return_dict=True,
                )

            # Extract hidden states
            if return_last_hidden_state and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_state = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]

                # Pool to fixed size representation
                if 'attention_mask' in inputs:
                    mask = inputs['attention_mask'].unsqueeze(-1)
                    pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    pooled = hidden_state.mean(dim=1)

                all_hidden_states.append(hidden_state[:, -1:, :])  # Last token
                all_pooled.append(pooled)

        # Stack batched outputs
        result = {}
        if all_pooled:
            result['pooled_output'] = torch.cat(all_pooled, dim=0)  # [B, hidden_dim]
            result['last_hidden_state'] = torch.cat(all_hidden_states, dim=0)  # [B, 1, hidden_dim]

        return result

    def _extract_outputs(
        self,
        outputs,
        inputs: Dict[str, torch.Tensor],
        return_last_hidden_state: bool,
    ) -> Dict[str, torch.Tensor]:
        """Extract relevant outputs from model forward pass.

        Matches the batched training path: returns last token as
        ``last_hidden_state`` [1, 1, hidden] and mean-pooled as
        ``pooled_output`` [1, hidden].
        """
        result = {}

        if return_last_hidden_state:
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None

            if hidden_states is not None:
                # Use last token only — consistent with _forward_batch
                result['last_hidden_state'] = hidden_states[:, -1:, :]

                # Create pooled output (mean of sequence)
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
