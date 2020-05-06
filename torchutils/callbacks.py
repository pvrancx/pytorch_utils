import os
from typing import Optional, Tuple, List, Dict, Any

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchutils.experiment import Experiment, DataLoaders, VALIDATION_LOSS_LABEL


class Callback:
    def __init__(self, priority: int = 0):
        self._priority = priority
        self.exp = None  # type: Optional[Experiment]
        self.data = None  # type: Optional[DataLoaders]
        self.last_batch = None  # type: Optional[Tuple]
        self.last_predictions = None  # type: Optional[torch.Tensor]

    @property
    def priority(self) -> int:
        return self._priority

    def get_state_dict(self) -> Dict[str, Any]:
        return {}

    def on_train_start(self, exp: Experiment, data: DataLoaders) -> bool:
        self.exp = exp
        self.data = data
        return True

    def on_epoch_start(self, epoch: int) -> bool:
        return True

    def on_epoch_end(self, epoch: int) -> bool:
        return True

    def on_batch_start(self, batch_id: int, batch: Tuple) -> bool:
        self.last_batch = batch
        return True

    def on_batch_end(self, batch_id: int, predictions: torch.Tensor, loss: float) -> bool:
        self.last_predictions = predictions
        return True

    def on_train_end(self) -> bool:
        return True


class CallbackHandler:
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self._callbacks = callbacks or []
        self._callbacks.sort(key=lambda x: x.priority)

    def get_state_dict(self):
        return {cb.__name__: (order, cb.get_state_dict()) for order, cb in self._callbacks}

    def _call_cb_method(self, method_name: str, **kwargs):
        for cb in self._callbacks:
            getattr(cb, method_name)(**kwargs)

    def add_callback(self, callback: Callback):
        self._callbacks.append(callback)
        self._callbacks.sort(key=lambda x: x.priority)

    def remove_callback(self, callback: Callback):
        self._callbacks = [item for item in self._callbacks if item[1] is not callback]

    def on_train_start(self, exp: Experiment, data: DataLoaders) -> bool:
        result = True
        for cb in self._callbacks: cb.on_train_start(exp, data)
        return result

    def on_epoch_start(self, epoch: int) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_epoch_start(epoch)
        return result

    def on_epoch_end(self, epoch: int) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_epoch_end(epoch)
        return result

    def on_batch_start(self, batch_id: int, batch: Tuple) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_batch_start(batch_id, batch)
        return result

    def on_batch_end(self, batch_id: int, predictions: torch.Tensor, loss: float) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_batch_end(batch_id, predictions, loss)
        return result

    def on_train_end(self) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_train_end()
        return result


class ModelSaverCallback(Callback):
    def __init__(self, save_path: str, frequency: int, priority: int = 1):
        super(ModelSaverCallback, self).__init__(priority)
        self._save_path = save_path
        self._freq = frequency

    def on_epoch_end(self, epoch: int) -> bool:
        if epoch % self._freq == 0:
            fname = os.path.join(self._save_path, "epoch_%d.chkpt" % epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.exp.model.state_dict(),
                'optimizer_state_dict': self.exp.optimizer.state_dict(),
                'loss': self.exp.metrics[epoch][VALIDATION_LOSS_LABEL],
            }, fname)
        return True


class LoggerCallback(Callback):
    def __init__(self, frequency: int = 20, alpha: float = 0.9, priority: int = 1):
        super(LoggerCallback, self).__init__(priority)
        self._freq = frequency
        self._avg = 0.
        self._alpha = alpha
        self._n_batches = 0

    def on_train_start(self, exp: Experiment, data: DataLoaders) -> bool:
        super(LoggerCallback, self).on_train_start(exp, data)
        self._n_batches = len(data.train)
        return True

    def on_batch_end(self, batch_id: int, predictions: torch.Tensor, loss: float) -> bool:
        if self._avg == 0.:
            self._avg = loss
        else:
            self._avg *= self._alpha
            self._avg += (1. - self._alpha) * loss

        if batch_id % self._freq == 0:
            print("Batch %d/%d - loss %1.5f" % (batch_id, self._n_batches,  self._avg))
        return True

    def on_epoch_end(self, epoch: int) -> bool:
        print("Finished epoch %d - metrics:" % epoch)
        for k, v in self.exp.metrics[epoch].items():
            print(k + ": %1.5f" % v)
        self._avg = 0.
        return True


class ScheduleStepper(Callback):
    def __init__(self, schedulers: List[Any], priority: int = 0):
        super(ScheduleStepper, self).__init__(priority)
        self._schedulers = schedulers

    def on_epoch_end(self, epoch: int) -> bool:
        for s in self._schedulers:
            if isinstance(s, ReduceLROnPlateau):
                s.step(metrics=self.exp.metrics[epoch][VALIDATION_LOSS_LABEL])
            else:
                s.step()
        return True





