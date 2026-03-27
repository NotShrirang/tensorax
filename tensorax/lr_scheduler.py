"""
Learning rate schedulers.
"""

import math
from .optim import Optimizer


class _LRScheduler:
    """Base class for all learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"Expected an Optimizer, got {type(optimizer).__name__}")
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        self._step_count = 0

        # Take the initial step so lr is set for epoch 0
        if last_epoch == -1:
            self.last_epoch = 0
            self._update_lr()

    def get_lr(self) -> float:
        """Compute the current learning rate. Override in subclasses."""
        raise NotImplementedError

    def step(self):
        """Advance to the next epoch and update the learning rate."""
        self._step_count += 1
        self.last_epoch += 1
        self._update_lr()

    def _update_lr(self):
        self.optimizer.lr = self.get_lr()

    def get_last_lr(self) -> float:
        """Return the last computed learning rate."""
        return self.optimizer.lr


class StepLR(_LRScheduler):
    """Decays the learning rate by ``gamma`` every ``step_size`` epochs.

    lr = base_lr * gamma ** (epoch // step_size)

    Args:
        optimizer: Wrapped optimizer.
        step_size: Period of learning rate decay.
        gamma: Multiplicative factor of learning rate decay. Default: 0.1.
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class ExponentialLR(_LRScheduler):
    """Decays the learning rate by ``gamma`` every epoch.

    lr = base_lr * gamma ** epoch

    Args:
        optimizer: Wrapped optimizer.
        gamma: Multiplicative factor of learning rate decay.
    """

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** self.last_epoch


class CosineAnnealingLR(_LRScheduler):
    """Cosine annealing schedule.

    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))

    Args:
        optimizer: Wrapped optimizer.
        T_max: Maximum number of epochs.
        eta_min: Minimum learning rate. Default: 0.
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.last_epoch / self.T_max)
        )


class LinearLR(_LRScheduler):
    """Linearly decays the learning rate over a given number of epochs.

    The multiplicative factor starts at ``start_factor`` and linearly changes
    to ``end_factor`` over ``total_iters`` epochs.

    Args:
        optimizer: Wrapped optimizer.
        start_factor: The number we multiply learning rate at the start. Default: 1.0/3.
        end_factor: The number we multiply learning rate at the end. Default: 1.0.
        total_iters: Number of iterations over which the factor changes. Default: 5.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch >= self.total_iters:
            return self.base_lr * self.end_factor
        t = self.last_epoch / max(self.total_iters, 1)
        factor = self.start_factor + (self.end_factor - self.start_factor) * t
        return self.base_lr * factor


class MultiStepLR(_LRScheduler):
    """Decays the learning rate by ``gamma`` once the epoch reaches one of
    the milestones.

    Args:
        optimizer: Wrapped optimizer.
        milestones: List of epoch indices (must be increasing).
        gamma: Multiplicative factor of learning rate decay. Default: 0.1.
    """

    def __init__(self, optimizer: Optimizer, milestones: list, gamma: float = 0.1, last_epoch: int = -1):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        n = sum(1 for m in self.milestones if m <= self.last_epoch)
        return self.base_lr * self.gamma ** n
