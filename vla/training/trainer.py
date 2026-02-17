"""
VLA Training Loop

Main trainer class for Vision-Language-Action model training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass, field
import json
import time
from tqdm import tqdm
import logging

from ..models import VLAModel
from .loss import VLALoss
from .scheduler import get_scheduler


@dataclass
class TrainerConfig:
    """Configuration for VLA training."""
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    min_lr: float = 1e-6

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 10

    # Optimization
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Device
    device: str = "cuda"

    # Logging
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_images: bool = True

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"


class VLATrainer:
    """
    Trainer for Vision-Language-Action models.

    Handles:
    - Training loop with mixed precision
    - Gradient accumulation
    - Checkpointing
    - Logging (tensorboard, wandb)
    - Evaluation
    """

    def __init__(
        self,
        model: VLAModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = get_scheduler(
            config.scheduler_type,
            self.optimizer,
            config.warmup_steps,
            total_steps,
            config.min_lr,
        )

        # Loss function
        self.loss_fn = VLALoss()

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        self.amp_dtype = getattr(torch, config.amp_dtype, torch.bfloat16)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Callbacks
        self.callbacks: List[Callable] = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter groups."""
        # Get trainable parameters
        params = self.model.get_trainable_parameters()

        # Create optimizer
        if self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        elif self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")

        return optimizer

    def _setup_logging(self) -> None:
        """Setup logging (tensorboard, wandb)."""
        # Setup basic logging
        logging.basicConfig(level=logging.INFO)

        # Wandb
        if self.config.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=vars(self.config),
                )
                self.use_wandb = True
            except ImportError:
                self.logger.warning("wandb not installed, skipping wandb logging")
                self.use_wandb = False
        else:
            self.use_wandb = False

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training metrics and best checkpoint path
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.config.num_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")

        training_stats = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
        }

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()
            training_stats["train_losses"].append(train_metrics["loss"])
            training_stats["learning_rates"].append(self.scheduler.get_last_lr()[0])

            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
                training_stats["val_losses"].append(val_metrics["loss"])

                # Early stopping check
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Epoch checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")

            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )

        # Final checkpoint
        self._save_checkpoint("final")

        return training_stats

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
                # Get instructions as list
                instructions = batch['instruction']
                if isinstance(instructions, str):
                    instructions = [instructions] * batch['context_image'].shape[0]

                outputs = self.model(
                    context_images=batch['context_image'],
                    wrist_images=batch['wrist_image'],
                    instructions=instructions,
                    proprio_state=batch.get('proprio_state'),
                    target_actions=batch['actions'],
                )

                loss = outputs['loss']
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    self._log_metrics({
                        "train/loss": loss.item() * self.config.gradient_accumulation_steps,
                        "train/lr": self.scheduler.get_last_lr()[0],
                    })

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Periodic checkpointing
            if self.global_step % self.config.save_every_n_steps == 0:
                self._save_checkpoint(f"step_{self.global_step}")

        return {"loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = self._move_to_device(batch)

            with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
                # Get instructions as list
                instructions = batch['instruction']
                if isinstance(instructions, str):
                    instructions = [instructions] * batch['context_image'].shape[0]

                outputs = self.model(
                    context_images=batch['context_image'],
                    wrist_images=batch['wrist_image'],
                    instructions=instructions,
                    proprio_state=batch.get('proprio_state'),
                    target_actions=batch['actions'],
                )

                loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

        val_loss = total_loss / max(num_batches, 1)

        self._log_metrics({"val/loss": val_loss})

        return {"loss": val_loss}

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to wandb and/or tensorboard."""
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=self.global_step)

    def _save_checkpoint(self, name: str, lightweight: bool = True) -> None:
        """
        Save model checkpoint.

        Args:
            name: Checkpoint name
            lightweight: If True, only save trainable weights (~100MB instead of ~16GB)
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        if lightweight:
            # Only save trainable components (LoRA, projector, action_expert, flow_head)
            # This reduces checkpoint size from ~16GB to ~100-200MB
            checkpoint = {
                "checkpoint_type": "lightweight",
                # Trainable model components
                "projector_state_dict": self.model.projector.state_dict(),
                "action_expert_state_dict": self.model.action_expert.state_dict(),
                "flow_head_state_dict": self.model.flow_head.state_dict(),
                # Normalization stats
                "action_mean": self.model.action_mean,
                "action_std": self.model.action_std,
                "proprio_mean": getattr(self.model, '_proprio_mean', None),
                "proprio_std": getattr(self.model, '_proprio_std', None),
                "dataset_action_mean": getattr(self.model, '_dataset_action_mean', None),
                "dataset_action_std": getattr(self.model, '_dataset_action_std', None),
                # Training state
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_val_loss": self.best_val_loss,
                "config": vars(self.config),
            }

            # Save LoRA weights if they exist
            if hasattr(self.model.vision_encoder, 'lora_modules'):
                checkpoint["lora_state_dict"] = self.model.vision_encoder.lora_modules.state_dict()
        else:
            # Full checkpoint (legacy mode)
            checkpoint = {
                "checkpoint_type": "full",
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_val_loss": self.best_val_loss,
                "config": vars(self.config),
            }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        # Log checkpoint size
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Saved checkpoint: {checkpoint_path} ({size_mb:.1f} MB)")

    def load_checkpoint(self, path: str) -> None:
        """
        Load checkpoint to resume training.

        Supports both lightweight (trainable weights only) and full checkpoints.
        """
        # Always load to CPU first to avoid OOM
        checkpoint = torch.load(path, map_location='cpu')

        checkpoint_type = checkpoint.get("checkpoint_type", "full")

        if checkpoint_type == "lightweight":
            # Load trainable components only
            self.model.projector.load_state_dict(checkpoint["projector_state_dict"])
            self.model.action_expert.load_state_dict(checkpoint["action_expert_state_dict"])
            self.model.flow_head.load_state_dict(checkpoint["flow_head_state_dict"])

            # Load action normalization stats
            if "action_mean" in checkpoint:
                self.model.action_mean = checkpoint["action_mean"]
            if "action_std" in checkpoint:
                self.model.action_std = checkpoint["action_std"]

            # Load LoRA weights if present
            if "lora_state_dict" in checkpoint and hasattr(self.model.vision_encoder, 'lora_modules'):
                self.model.vision_encoder.lora_modules.load_state_dict(checkpoint["lora_state_dict"])

            self.logger.info(f"Loaded lightweight checkpoint from {path}")
        else:
            # Full checkpoint (legacy) - extract only trainable weights to avoid OOM
            full_state = checkpoint["model_state_dict"]

            # Extract and load projector weights
            projector_state = {k.replace("projector.", ""): v for k, v in full_state.items() if k.startswith("projector.")}
            if projector_state:
                self.model.projector.load_state_dict(projector_state)

            # Extract and load action_expert weights
            action_expert_state = {k.replace("action_expert.", ""): v for k, v in full_state.items() if k.startswith("action_expert.")}
            if action_expert_state:
                self.model.action_expert.load_state_dict(action_expert_state)

            # Extract and load flow_head weights
            flow_head_state = {k.replace("flow_head.", ""): v for k, v in full_state.items() if k.startswith("flow_head.")}
            if flow_head_state:
                self.model.flow_head.load_state_dict(flow_head_state)

            # Extract and load LoRA weights
            if hasattr(self.model.vision_encoder, 'lora_modules'):
                lora_state = {k.replace("vision_encoder.lora_modules.", ""): v for k, v in full_state.items() if "lora_modules" in k}
                if lora_state:
                    self.model.vision_encoder.lora_modules.load_state_dict(lora_state)

            # Load normalization stats
            if "action_mean" in full_state:
                self.model.action_mean = full_state["action_mean"]
            if "action_std" in full_state:
                self.model.action_std = full_state["action_std"]

            # Clear full_state to free memory
            del full_state
            del checkpoint["model_state_dict"]
            import gc
            gc.collect()

            self.logger.info(f"Loaded trainable weights from full checkpoint: {path}")

        # Load training state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    def add_callback(self, callback: Callable) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)


def train_vla(
    model: VLAModel,
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    config: Optional[TrainerConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train a VLA model.

    Args:
        model: VLA model to train
        train_data_dir: Path to training data
        val_data_dir: Path to validation data (optional)
        config: Training configuration
        **kwargs: Override config parameters

    Returns:
        Training statistics
    """
    from ..data import VLADataset, create_dataloader

    # Create config
    if config is None:
        config = TrainerConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create data loaders
    train_loader = create_dataloader(
        train_data_dir,
        batch_size=config.batch_size,
        num_workers=4,
    )

    val_loader = None
    if val_data_dir:
        val_loader = create_dataloader(
            val_data_dir,
            batch_size=config.batch_size,
            num_workers=4,
            augment=False,
        )

    # Create trainer
    trainer = VLATrainer(model, config, train_loader, val_loader)

    # Train
    return trainer.train()
