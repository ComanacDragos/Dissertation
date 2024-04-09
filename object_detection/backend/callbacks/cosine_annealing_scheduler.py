import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np


class CosineAnnealingScheduler(Callback):
    def __init__(self, n_min, n_max, T):
        super(CosineAnnealingScheduler, self).__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.T = T
        self.learning_rates = {
            # 0: 1e-3
        }

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(logs.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if epoch not in self.learning_rates:
            scheduled_lr = self.n_min + (1 / 2) * (self.n_max - self.n_min) * (1 + np.cos(epoch / self.T * np.pi))
        else:
            scheduled_lr = self.learning_rates[epoch]
        tf.keras.backend.set_value(logs.model.optimizer.lr, scheduled_lr)
        print(f"\nEpoch {epoch}: Learning rate is {scheduled_lr}.")
