from typing import Dict, Callable, List, Optional, Tuple

import torch

from torchutils.callbacks import Callback
from torchutils.experiment import Experiment


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    acc = (targets == predictions.argmax(-1)).float()
    return torch.mean(acc).detach()


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean((predictions - targets) ** 2.).detach()


class BatchMetric(Callback):
    """Metric that computed over batches"""
    def __init__(
            self,
            f: Callable,
            name: str = None,
            f_args: Dict = None,
            train: bool = True,
            priority: int = -1
    ):
        super(BatchMetric, self).__init__(priority, name or f.__name__)
        self._fun = f
        self._f_args = f_args or {}
        self._total = 0.
        self._count = 0.
        self._train = train

    def on_epoch_start(self, **kwargs):
        self._total = 0.
        self._count = 0.

    def on_epoch_end(self, epoch_id: int, experiment: Experiment, **kwargs):
        value = self._total / self._count
        experiment.log_epoch_metric(epoch_id, self.name, value)

    def on_batch_end(
            self,
            last_batch: Tuple,
            batch_predictions: torch.Tensor,
            batch_loss: float,
            training: bool,
            **kwargs):
        n_items = batch_predictions.shape[0]
        if (self._train and training) or (not self._train and not training):
            self._count += n_items
            self._total += self._fun(batch_predictions, last_batch[-1], **self._f_args) * n_items
