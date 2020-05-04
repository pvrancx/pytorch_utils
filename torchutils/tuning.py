from typing import Tuple

import matplotlib.pyplot as plt
import torch

from torchutils.experiment import Experiment, DataLoaders
from torchutils.train import process_batch
from torchutils.utils import smooth


def update_lr(optimizer: torch.optim.Optimizer, new_lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def scan_lr(
        exp: Experiment,
        data: DataLoaders,
        min_lr: float,
        max_lr: float,
        n_epochs: int
) -> Tuple:

    """
    Scan over learning rates.
    Smith, 2015
    """
    update_lr(exp.optimizer, min_lr)
    total_batches = len(data.train) * n_epochs - 1
    lrs = []
    train_loss = []
    n_batches = 0
    for epoch in range(n_epochs):
        for batch in data.train:
            new_lr = min_lr + (n_batches / total_batches) * (max_lr - min_lr)
            update_lr(exp.optimizer, new_lr)
            train_loss.append(process_batch(exp, batch))
            lrs.append(new_lr)
            n_batches += 1
    return lrs, train_loss


def lr_plot(lrs, train_loss):
    plt.figure()
    plt.plot(lrs, smooth(train_loss, 0.99), label='train loss')
    plt.legend()
    plt.show()


