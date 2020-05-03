from typing import Tuple

import torch

from torchutils.callbacks import CallbackHandler
from torchutils.experiment import Experiment, DataLoaders


def process_batch(exp: Experiment, batch: Tuple) -> float:
    inputs, labels = batch
    exp.optimizer.zero_grad()
    outputs = exp.model(inputs)
    loss = exp.loss_fn(outputs, labels)
    loss.backward()
    exp.optimizer.step()
    return loss


def train(
        exp: Experiment,
        data_loader: torch.utils.data.DataLoader,
        callbacks: CallbackHandler
) -> float:

    exp.model.to(exp.config.device)
    exp.model.train()
    train_loss = 0.0
    count = 0
    for batch_id, (inputs, labels) in enumerate(data_loader):
        callbacks.on_batch_start(batch_id, (inputs, labels))
        inputs, labels = inputs.to(exp.config.device), labels.to(exp.config.device)
        loss = process_batch(exp, (inputs, labels))
        train_loss += loss
        count += labels.size(0)
        callbacks.on_batch_end(batch_id, loss)
    return train_loss / len(data_loader)


def evaluate(
        exp: Experiment,
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
            test_loss += exp.loss_fn(outputs, labels).sum()
            count += labels.size(0)
    return test_loss / count


def fit(exp: Experiment, data: DataLoaders, callbacks: CallbackHandler):
    callbacks.on_train_start(exp, data)
    for epoch in range(exp.config.max_epochs):
        callbacks.on_epoch_start(epoch)
        train(exp, data.train, callbacks)
        valid_loss = evaluate(exp, data.test)
        if exp.lr_scheduler is not None:
            exp.lr_scheduler.step(metrics=valid_loss)
        callbacks.on_epoch_end(epoch, valid_loss)
    callbacks.on_train_end()
    return exp
