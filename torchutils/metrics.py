from typing import Dict, Callable

import torch

from torchutils.callbacks import Callback
from torchutils.experiment import Experiment


def evaluate(
        exp: Experiment,
        loss_fn: Callable,
        data_loader: torch.utils.data.DataLoader,
) -> float:

    exp.model.to(exp.config.device)
    exp.model.eval()
    test_loss = 0.0
    count = 0
    for inputs, labels in data_loader:
        with torch.no_grad():
            inputs, labels = inputs.to(exp.config.device), labels.to(exp.config.device)
            outputs = exp.model(inputs)
            test_loss += loss_fn(outputs, labels)
            count += labels.size(0)
    return test_loss / count


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    acc = (targets.argmax(-1) == predictions.argmax(-1)).float()
    return torch.mean(acc).detach().numpy()


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean((predictions - targets) ** 2.).detach().numpy()


class BatchMetric(Callback):
    """Metric that computed over batches"""
    def __init__(self, f: Callable, name: str = None, f_args: Dict = None):
        super(BatchMetric, self).__init__()
        self._name = name or f.__name__
        self._fun = f
        self._f_args = f_args or {}
        self._total = 0.
        self._count = 0.

    def on_epoch_start(self, epoch: int) -> bool:
        self._total = 0.
        self._count = 0.
        return True

    def on_epoch_end(self, epoch: int) -> bool:
        value = self._total / self._count
        if epoch in self.exp.metrics:
            self.exp.metrics[epoch].update({self._name: value})
        else:
            self.exp.metrics[epoch] = {self._name: value}
        return True

    def on_batch_end(self, batch_id: int, predictions: torch.Tensor, loss: float) -> bool:
        n_items = predictions.shape[0]
        self._count += n_items
        self._total += self._fun(predictions, self.last_batch[-1], **self._f_args) * n_items
        return True


class ValidationMetric(Callback):
    """ Metric that evaluates criterion on given dataset at end of epoch"""
    def __init__(
            self,
            criterion: Callable,
            dataloader: torch.utils.data.DataLoader,
            name: str = None
    ):
        super(ValidationMetric, self). __init__()
        self.criterion = criterion
        self.dataloader = dataloader
        self._name = name or criterion.__name__

    def on_epoch_end(self, epoch: int) -> bool:
        value = evaluate(self.exp, self.criterion, self.dataloader)
        self.exp.metrics[epoch] = {self._name: value}
        return True
