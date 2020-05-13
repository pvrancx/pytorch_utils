import os
from typing import Optional, Tuple, List, Dict, Any

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from torchutils.experiment import Experiment, DataLoaders, VALIDATION_LOSS_LABEL


class Callback:
    def __init__(self, priority: int = 0, name: str = None):
        self._priority = priority
        self._name = name or self.__class__.__name__

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def name(self) -> str:
        return self._name

    def get_state_dict(self) -> Dict[str, Any]:
        return {}

    def on_train_start(self, **kwargs) -> Optional[Dict[str, Any]]:
        pass

    def on_epoch_start(self, **kwargs) -> Optional[Dict[str, Any]]:
        pass

    def on_epoch_end(self, **kwargs) -> Optional[Dict[str, Any]]:
        pass

    def on_batch_start(self, **kwargs) -> Optional[Dict[str, Any]]:
        pass

    def on_batch_end(self, **kwargs) -> Optional[Dict[str, Any]]:
        pass

    def on_train_end(self, **kwargs) -> Optional[Dict[str, Any]]:
        pass


class CallbackHandler:
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self._callbacks = callbacks or []
        self._callbacks.sort(key=lambda x: x.priority)
        self._state = {}

    def _call_cb_method(self, method_name: str):
        for cb in self._callbacks:
            cb_result = getattr(cb, method_name)(**self._state) or {}
            self._state.update(cb_result)

    def add_callback(self, callback: Callback):
        self._callbacks.append(callback)
        self._callbacks.sort(key=lambda x: x.priority)

    def remove_callback(self, callback: Callback):
        self._callbacks = [item for item in self._callbacks if item[1] is not callback]

    def on_train_start(self, exp: Experiment, data: DataLoaders):
        self._state['experiment'] = exp
        self._state['data'] = data
        self._call_cb_method('on_train_start')

    def on_epoch_start(self, epoch: int):
        self._state['epoch_id'] = epoch
        self._call_cb_method('on_epoch_start')

    def on_epoch_end(self):
        self._call_cb_method('on_epoch_end')

    def on_batch_start(self, batch_id: int, batch: Tuple, train: bool):
        self._state['last_batch'] = batch
        self._state['batch_id'] = batch_id
        self._state['training'] = train
        self._call_cb_method('on_batch_start')

    def on_batch_end(self, predictions: torch.Tensor, loss: torch.Tensor):
        self._state['batch_predictions'] = predictions
        self._state['batch_loss'] = loss
        self._call_cb_method('on_batch_end')

    def on_train_end(self):
        self._call_cb_method('on_train_end')


class ModelSaverCallback(Callback):
    def __init__(self, save_path: str, frequency: int, improve_only: bool = True, priority: int = 1):
        super(ModelSaverCallback, self).__init__(priority)
        self._save_path = save_path
        self._freq = frequency
        self._improve = improve_only
        self._best = float("Inf")

    def on_epoch_end(self, epoch_id: int, experiment: Experiment, **kwargs):
        loss = experiment.epoch_metrics(epoch_id)[VALIDATION_LOSS_LABEL]
        should_save = False
        if self._improve:
            if loss < self._best:
                self._best = loss
                should_save = True
        else:
            should_save = epoch_id % self._freq == 0

        if should_save:
            fname = os.path.join(self._save_path, "epoch_%d.chkpt" % epoch_id)
            torch.save({
                'epoch': epoch_id,
                'model_state_dict': experiment.model.state_dict(),
                'optimizer_state_dict': experiment.optimizer.state_dict(),
                'loss': loss,
            }, fname)


class LoggerCallback(Callback):
    def __init__(self, frequency: int = 20, alpha: float = 0.9, priority: int = 1):
        super(LoggerCallback, self).__init__(priority)
        self._freq = frequency
        self._avg = 0.
        self._alpha = alpha
        self._n_batches = 0

    def on_train_start(self, data: DataLoaders, **kwargs):
        self._n_batches = len(data.train)

    def on_batch_end(self, batch_id: int, batch_loss: float, training: bool, **kwargs):
        if not training:
            return

        if self._avg == 0.:
            self._avg = batch_loss
        else:
            self._avg *= self._alpha
            self._avg += (1. - self._alpha) * batch_loss

        if batch_id % self._freq == 0:
            print("Batch %d/%d - loss %1.5f" % (batch_id, self._n_batches,  self._avg))

    def on_epoch_end(self, epoch_id: int, experiment: Experiment, **kwargs):
        print("Finished epoch %d - metrics:" % epoch_id)
        for k, v in experiment.epoch_metrics(epoch_id).items():
            print(k + ": %1.5f" % v)
        self._avg = 0.


class ScheduleStepper(Callback):
    def __init__(self, schedulers: List[Any], priority: int = 0):
        super(ScheduleStepper, self).__init__(priority)
        self._schedulers = schedulers

    def on_epoch_end(self, epoch_id: int, experiment: Experiment, **kwargs):
        for s in self._schedulers:
            if isinstance(s, ReduceLROnPlateau):
                s.step(metrics=experiment.metrics[epoch_id][VALIDATION_LOSS_LABEL])
            else:
                s.step()


class TensorBoardLogger(Callback):
    def __init__(self, path: str, priority: int = 1):
        super(TensorBoardLogger, self).__init__(priority)
        self._writer = SummaryWriter(log_dir=path)

    def on_train_start(self, experiment: Experiment, data: DataLoaders, **kwargs):
        self._writer.add_graph(experiment.model, next(iter(data.train))[0])

    def on_epoch_end(self, epoch_id: int, experiment: Experiment, **kwargs):
        for k, v in experiment.metrics[epoch_id].items():
            self._writer.add_scalar(k, v, epoch_id)

    def on_train_end(self, **kwargs):
        self._writer.close()





