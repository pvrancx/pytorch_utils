from typing import Tuple, Callable

import matplotlib.pyplot as plt
import torch

from torchutils.experiment import Experiment, DataLoaders
from torchutils.train import process_batch
from torchutils.utils import smooth, linear_annealing


def update_lr(optimizer: torch.optim.Optimizer, new_lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def scan_lr(
        exp: Experiment,
        data: DataLoaders,
        min_lr: float,
        max_lr: float,
        n_epochs: int,
        anneal_f: Callable[[float, float, float], float] = linear_annealing
) -> Tuple:

    """
    Scan learning rate range.
    Smith, 2015
    """
    exp.model.to(exp.config.device)
    exp.model.train()

    update_lr(exp.optimizer, min_lr)
    total_batches = len(data.train) * n_epochs - 1
    lrs = []
    train_loss = []
    n_batches = 0
    for epoch in range(n_epochs):
        for batch in data.train:
            new_lr = anneal_f(min_lr, max_lr, n_batches / total_batches)
            update_lr(exp.optimizer, new_lr)
            train_loss.append(process_batch(exp, batch)[-1].detach().numpy())
            lrs.append(new_lr)
            n_batches += 1
    return lrs, train_loss


def lr_plot(lrs, train_loss):
    plt.figure()
    smoothed_loss = smooth(train_loss, 0.99)
    min_val = min(smoothed_loss)
    plt.ylim([min_val, min_val * 3])
    plt.plot(lrs, smoothed_loss, label='train loss')
    plt.legend()
    plt.show()


