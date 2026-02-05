"""
Learning rate schedulers for VLA training
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, List


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.

    Common schedule for transformer training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and linear decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Linear decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            scale = 1.0 - progress

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class WarmupConstantScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup then constant LR.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            scale = 1.0

        return [base_lr * scale for base_lr in self.base_lrs]


class CyclicCosineScheduler(_LRScheduler):
    """
    Cyclic cosine annealing with warm restarts.

    Based on SGDR: Stochastic Gradient Descent with Warm Restarts.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cycle_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
        cycle_mult: float = 1.0,
        last_epoch: int = -1,
    ):
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.cycle_mult = cycle_mult

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch

        # Warmup phase
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        # Find current cycle
        step_in_schedule = step - self.warmup_steps
        cycle_length = self.cycle_steps
        cycle = 0

        while step_in_schedule >= cycle_length:
            step_in_schedule -= cycle_length
            cycle_length = int(cycle_length * self.cycle_mult)
            cycle += 1

        # Cosine within cycle
        progress = step_in_schedule / cycle_length
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
    **kwargs,
) -> _LRScheduler:
    """
    Factory function to create learning rate scheduler.

    Args:
        name: Scheduler name ("cosine", "linear", "constant", "cyclic")
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Learning rate scheduler
    """
    schedulers = {
        "cosine": WarmupCosineScheduler,
        "linear": WarmupLinearScheduler,
        "constant": WarmupConstantScheduler,
        "cyclic": CyclicCosineScheduler,
    }

    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")

    scheduler_cls = schedulers[name]

    if name == "constant":
        return scheduler_cls(optimizer, warmup_steps)
    elif name == "cyclic":
        cycle_steps = kwargs.get("cycle_steps", total_steps // 4)
        cycle_mult = kwargs.get("cycle_mult", 1.0)
        return scheduler_cls(optimizer, cycle_steps, warmup_steps, min_lr, cycle_mult)
    else:
        return scheduler_cls(optimizer, warmup_steps, total_steps, min_lr)


class GradualUnfreezeScheduler:
    """
    Scheduler for gradually unfreezing model layers during training.

    Useful for fine-tuning VLM with progressive unfreezing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        unfreeze_schedule: dict,  # {step: [layer_names]}
    ):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        self._unfrozen_steps = set()

    def step(self, current_step: int) -> List[str]:
        """
        Check and apply unfreezing for current step.

        Args:
            current_step: Current training step

        Returns:
            List of layer names that were unfrozen
        """
        unfrozen = []

        for step, layers in self.unfreeze_schedule.items():
            if step <= current_step and step not in self._unfrozen_steps:
                for layer_name in layers:
                    self._unfreeze_layer(layer_name)
                    unfrozen.append(layer_name)
                self._unfrozen_steps.add(step)

        return unfrozen

    def _unfreeze_layer(self, layer_name: str) -> None:
        """Unfreeze parameters matching layer name."""
        for name, param in self.model.named_parameters():
            if layer_name in name:
                param.requires_grad = True
