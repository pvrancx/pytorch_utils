from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict

import torch
import torch.nn as nn

VALIDATION_LOSS_LABEL = 'validation loss'


@dataclass
class Config:
    max_epochs: int = 200
    device: torch.device = torch.device('cpu')


@dataclass
class DataLoaders:
    train: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


@dataclass
class Experiment:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable
    config: Config
    metrics: Dict[int, Dict] = field(default_factory=lambda: defaultdict(dict))

    def log_epoch_metric(self, epoch: int, name: str, value: float):
        self.metrics[epoch].update({name: value})

    def epoch_metrics(self, epoch: int):
        return self.metrics[epoch]

