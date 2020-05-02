from typing import NamedTuple, Callable
import torch
import torch.nn as nn


class Config(NamedTuple):
    max_epochs: int
    device: torch.device


class DataLoaders(NamedTuple):
    train: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


class Experiment(NamedTuple):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable
    config: Config

