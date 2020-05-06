from typing import Tuple, Optional, List, Callable, Any

import torch

from torchutils.callbacks import CallbackHandler, Callback, ScheduleStepper
from torchutils.experiment import Experiment, DataLoaders, VALIDATION_LOSS_LABEL
from torchutils.metrics import ValidationMetric, BatchMetric


def process_batch(exp: Experiment, batch: Tuple) -> Tuple[torch.Tensor, float]:
    inputs, labels = batch[0].to(exp.config.device), batch[1].to(exp.config.device)
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
        predictions, loss = process_batch(exp, (inputs, labels))
        callbacks.on_batch_end(batch_id, predictions, loss)


def _create_callbacks(
        exp: Experiment,
        data: DataLoaders,
        metrics: Optional[List[Callable]] = None,
        callbacks: Optional[List[Callback]] = None,
        schedulers: Optional[List[Any]] = None
) -> CallbackHandler:

    schedulers = schedulers or []
    metrics = metrics or []
    callbacks = callbacks or []
    metric_cbs = [BatchMetric(metric) for metric in metrics]
    all_callbacks = metric_cbs + callbacks
    handler = CallbackHandler(all_callbacks)
    priorities = [cb.priority for cb in all_callbacks]
    min_priority, max_priorities = min(priorities) - 1, max(priorities) + 1

    handler.add_callback(ValidationMetric(
        exp.loss_fn, data.test, VALIDATION_LOSS_LABEL, min_priority)
    )
    handler.add_callback(ScheduleStepper(schedulers, max_priorities))
    return handler


def fit(
        exp: Experiment,
        data: DataLoaders,
        metrics: Optional[List[Callable]] = None,
        callbacks: Optional[List[Callback]] = None,
        lr_schedulers: Optional[List[Any]] = None
):

    cb_handler = _create_callbacks(exp, data, metrics, callbacks, lr_schedulers)
    cb_handler.on_train_start(exp, data)
    for epoch in range(exp.config.max_epochs):
        cb_handler.on_epoch_start(epoch)
        train(exp, data.train, cb_handler)
        cb_handler.on_epoch_end(epoch)
    cb_handler.on_train_end()
    return exp
