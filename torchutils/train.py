from typing import Tuple, Optional, List, Callable, Any, Dict

import torch

from torchutils.callbacks import CallbackHandler, Callback, ScheduleStepper
from torchutils.experiment import Experiment, DataLoaders, VALIDATION_LOSS_LABEL
from torchutils.metrics import BatchMetric


def get_batch_loss(exp: Experiment, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = batch[0].to(exp.config.device), batch[1].to(exp.config.device)
    outputs = exp.model(inputs)
    loss = exp.loss_fn(outputs, labels)
    return outputs, loss


def train_batch(exp: Experiment, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    exp.optimizer.zero_grad()
    outputs, loss = get_batch_loss(exp, batch)
    loss.backward()
    exp.optimizer.step()
    return outputs, loss


def train(
        exp: Experiment,
        data_loader: torch.utils.data.DataLoader,
        callbacks: CallbackHandler
):

    exp.model.to(exp.config.device)
    exp.model.train()
    for batch_id, (inputs, labels) in enumerate(data_loader):
        callbacks.on_batch_start(batch_id, (inputs, labels), True)
        predictions, loss = train_batch(exp, (inputs, labels))
        callbacks.on_batch_end(predictions, loss)


def validate(
        exp: Experiment,
        data_loader: torch.utils.data.DataLoader,
        callbacks: CallbackHandler
) -> float:

    exp.model.to(exp.config.device)
    exp.model.eval()
    test_loss = 0.0
    count = 0
    for batch_id, (inputs, labels) in enumerate(data_loader):
        callbacks.on_batch_start(batch_id, (inputs, labels), False)
        with torch.no_grad():
            outputs, test_loss = get_batch_loss(exp, (inputs, labels))
        count += 1
        callbacks.on_batch_end(outputs, test_loss)
    return test_loss / count


def _create_callbacks(
        exp: Experiment,
        train_metrics: Optional[Optional[Dict[str, Callable]]] = None,
        validation_metrics: Optional[Optional[Dict[str, Callable]]] = None,
        callbacks: Optional[List[Callback]] = None,
        schedulers: Optional[List[Any]] = None
) -> CallbackHandler:

    schedulers = schedulers or []
    train_metrics = train_metrics or {}
    val_metrics = validation_metrics or {}
    callbacks = callbacks or []
    train_metric_cbs = [BatchMetric(f=v, name=k, train=True) for k, v in train_metrics.items()]
    val_metric_cbs = [BatchMetric(f=v, name=k, train=False) for k, v in val_metrics.items()]
    all_callbacks = val_metric_cbs + train_metric_cbs + callbacks
    handler = CallbackHandler(all_callbacks)
    priorities = [cb.priority for cb in all_callbacks]
    min_priority, max_priorities = min(priorities) - 1, max(priorities) + 1

    handler.add_callback(BatchMetric(f=exp.loss_fn, name=VALIDATION_LOSS_LABEL, train=False))
    handler.add_callback(ScheduleStepper(schedulers, max_priorities))
    return handler


def fit(
        exp: Experiment,
        data: DataLoaders,
        train_metrics: Optional[Dict[str, Callable]] = None,
        validation_metrics: Optional[Dict[str, Callable]] = None,
        callbacks: Optional[List[Callback]] = None,
        lr_schedulers: Optional[List[Any]] = None
):

    cb_handler = _create_callbacks(
        exp, train_metrics, validation_metrics, callbacks, lr_schedulers
    )
    cb_handler.on_train_start(exp, data)
    for epoch in range(exp.config.max_epochs):
        cb_handler.on_epoch_start(epoch)
        train(exp, data.train, cb_handler)
        if data.test is not None:
            validate(exp, data.test, cb_handler)
        cb_handler.on_epoch_end()
    cb_handler.on_train_end()
    return exp
