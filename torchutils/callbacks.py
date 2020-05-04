import os
from typing import Optional, Tuple, List

import torch

from torchutils.experiment import Experiment, DataLoaders


class Callback:
    def __init__(self):
        self.exp = None  # type: Optional[Experiment]
        self.data = None  # type: Optional[DataLoaders]
        self.last_batch = None  # type: Optional[Tuple]
        self.last_predictions = None  # type: Optional[torch.Tensor]

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
        self._callbacks = [(0, cb) for cb in callbacks] if callbacks is not None else []
        self._callbacks.sort(key=lambda x: x[0])

    def add_callback(self, callback: Callback, priority: int = 0):
        self._callbacks.append((priority, callback))
        self._callbacks.sort(key=lambda x: x[0])

    def remove_callback(self, callback: Callback):
        self._callbacks = [item for item in self._callbacks if item[1] is not callback]

    def on_train_start(self, exp: Experiment, data: DataLoaders) -> bool:
        result = True
        for _, cb in self._callbacks: cb.on_train_start(exp, data)
        return result

    def on_epoch_start(self, epoch: int) -> bool:
        result = True
        for _, cb in self._callbacks: result = result and cb.on_epoch_start(epoch)
        return result

    def on_epoch_end(self, epoch: int) -> bool:
        result = True
        for _, cb in self._callbacks: result = result and cb.on_epoch_end(epoch)
        return result

    def on_batch_start(self, batch_id: int, batch: Tuple) -> bool:
        result = True
        for _, cb in self._callbacks: result = result and cb.on_batch_start(batch_id, batch)
        return result

    def on_batch_end(self, batch_id: int, predictions: torch.Tensor, loss: float) -> bool:
        result = True
        for _, cb in self._callbacks: result = result and cb.on_batch_end(batch_id, predictions, loss)
        return result

    def on_train_end(self) -> bool:
        result = True
        for _, cb in self._callbacks: result = result and cb.on_train_end()
        return result


class ModelSaverCallback(Callback):
    def __init__(self, save_path: str, frequency: int):
        super(ModelSaverCallback, self).__init__()
        self._save_path = save_path
        self._freq = frequency

    def on_epoch_end(self, epoch: int) -> bool:
        if epoch % self._freq == 0:
            fname = os.path.join(self._save_path, "epoch_%d.chkpt" % epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.exp.model.state_dict(),
                'optimizer_state_dict': self.exp.optimizer.state_dict(),
                'loss': self.exp.metrics[epoch]['validation_loss'],
            }, fname)
        return True


class LoggerCallback(Callback):
    def __init__(self, frequency: int = 20, alpha=0.9):
        super(LoggerCallback, self).__init__()
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
        loss = self.exp.metrics[epoch]['validation_loss']
        print("Finished epoch %d - loss %1.5f" % (epoch, loss))
        self._avg = 0.
        return True
