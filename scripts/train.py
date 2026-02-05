#!/usr/bin/env python3
"""
VLA Training Script

Train a Vision-Language-Action model on robot demonstration data.

Usage:
    python scripts/train.py --config configs/training/pretrain_config.yaml
    python scripts/train.py --data_dir /path/to/demos --epochs 100
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vla.models import VLAModel, create_vla_model
from vla.models.vla_model import VLAConfig
from vla.models.vision_encoder import VisionEncoderConfig
from vla.models.projector import ProjectorConfig
from vla.models.action_expert import ActionExpertConfig
from vla.models.flow_matching import FlowMatchingConfig
from vla.training import VLATrainer, TrainerConfig
from vla.data import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLA model")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to training data directory")
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Path to validation data directory")

    # Model
    parser.add_argument("--vlm_model", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Vision-language model name")
    parser.add_argument("--action_dim", type=int, default=7,
                        help="Robot action dimension")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="Action chunk size")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for action expert")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps")

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name")
    parser.add_argument("--wandb_run", type=str, default=None,
                        help="Wandb run name")

    # Config file (overrides CLI args)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    return parser.parse_args()


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        file_config = load_config_file(args.config)
        # Override args with file config
        for key, value in file_config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    print("=" * 60)
    print("VLA Training")
    print("=" * 60)
    print(f"VLM Model: {args.vlm_model}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Action Dimension: {args.action_dim}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    # Create model
    print("\nCreating VLA model...")
    model = create_vla_model(
        model_name=args.vlm_model,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        freeze_vlm=True,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        device_map="auto",
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Create data loaders
    print("\nLoading training data...")
    train_loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        augment=True,
    )
    print(f"Training samples: {len(train_loader.dataset)}")

    val_loader = None
    if args.val_data_dir:
        print("Loading validation data...")
        val_loader = create_dataloader(
            args.val_data_dir,
            batch_size=args.batch_size,
            num_workers=4,
            chunk_size=args.chunk_size,
            action_dim=args.action_dim,
            augment=False,
        )
        print(f"Validation samples: {len(val_loader.dataset)}")

    # Create trainer config
    trainer_config = TrainerConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )

    # Create trainer
    trainer = VLATrainer(
        model=model,
        config=trainer_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    stats = trainer.train()

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
