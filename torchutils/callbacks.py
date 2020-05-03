import os
from typing import Optional, Tuple, List

import torch

from torchutils.experiment import Experiment, DataLoaders


class Callback:
    def __init__(self):
        self.exp = None  # type: Optional[Experiment]

    def on_train_start(self, exp: Experiment, data: DataLoaders) -> bool:
        self.exp = exp
        return True

    def on_epoch_start(self, epoch: int) -> bool:
        return True

    def on_epoch_end(self, epoch: int, loss: float) -> bool:
        return True

    def on_batch_start(self, batch_id: int, batch: Tuple) -> bool:
        return True

    def on_batch_end(self, batch_id: int, loss: float) -> bool:
        return True

    def on_train_end(self) -> bool:
        return True


class CallbackHandler:
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self._callbacks = callbacks or []

    def add_callback(self, callback: Callback):
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callback):
        self._callbacks.remove(callback)

    def on_train_start(self, exp: Experiment, data: DataLoaders) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_train_start(exp, data)
        return result

    def on_epoch_start(self, epoch: int) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_epoch_start(epoch)
        return result

    def on_epoch_end(self, epoch: int, loss: float) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_epoch_end(epoch, loss)
        return result

    def on_batch_start(self, batch_id: int, batch: Tuple) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_batch_start(batch_id, batch)
        return result

    def on_batch_end(self, batch_id: int, loss: float) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_batch_end(batch_id, loss)
        return result

    def on_train_end(self) -> bool:
        result = True
        for cb in self._callbacks: result = result and cb.on_train_end()
        return result


class ModelSaverCallback(Callback):
    def __init__(self, save_path: str, frequency: int):
        super(ModelSaverCallback, self).__init__()
        self._save_path = save_path
        self._freq = frequency

    def on_epoch_end(self, epoch: int, loss: float) -> bool:
        if epoch % self._freq == 0:
            fname = os.path.join(self._save_path, "epoch_%d.chkpt" % epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.exp.model.state_dict(),
                ' optimizer_state_dict': self.exp.optimizer.state_dict(),
                'loss': loss,
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
        self._n_batches = len(data.train)

    def on_batch_end(self, batch_id: int, loss: float) -> bool:
        self._avg = loss if self._avg == 0. \
            else (self._avg * self._alpha) + (1. - self._alpha) * loss
        if batch_id % self._freq == 0:
            print("Batch %d/%d - loss %1.5f" % (batch_id, self._n_batches,  self._avg))
        return True

    def on_epoch_end(self, epoch: int, loss: float) -> bool:
        print("Finished epoch %d - loss %1.5f" % (epoch, loss))
        self._avg = 0.
        return True
