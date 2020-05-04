from typing import Tuple, Callable

import torch

from torchutils.callbacks import CallbackHandler
from torchutils.experiment import Experiment, DataLoaders
from torchutils.metrics import ValidationMetric


def process_batch(exp: Experiment, batch: Tuple) -> Tuple[torch.Tensor, float]:
    inputs, labels = batch
    exp.optimizer.zero_grad()
    outputs = exp.model(inputs)
    loss = exp.loss_fn(outputs, labels)
    loss.backward()
    exp.optimizer.step()
    return outputs, loss


def train(
        exp: Experiment,
        data_loader: torch.utils.data.DataLoader,
        callbacks: CallbackHandler
) -> float:

    exp.model.to(exp.config.device)
    exp.model.train()
    for batch_id, (inputs, labels) in enumerate(data_loader):
        callbacks.on_batch_start(batch_id, (inputs, labels))
        inputs, labels = inputs.to(exp.config.device), labels.to(exp.config.device)
        predictions, loss = process_batch(exp, (inputs, labels))
        callbacks.on_batch_end(batch_id, predictions, loss)


def fit(exp: Experiment, data: DataLoaders, callbacks: CallbackHandler):
    callbacks.add_callback(ValidationMetric(exp.loss_fn, data.test,  'validation_loss'), -1)
    callbacks.on_train_start(exp, data)
    for epoch in range(exp.config.max_epochs):
        callbacks.on_epoch_start(epoch)
        train(exp, data.train, callbacks)
        callbacks.on_epoch_end(epoch)
        if exp.lr_scheduler is not None:
            exp.lr_scheduler.step(metrics=exp.metrics[epoch]['validation_loss'])
    callbacks.on_train_end()
    return exp
