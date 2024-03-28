import os
from pathlib import Path

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot
from tensorflow.keras.callbacks import Callback
import numpy as np
from cv2 import imwrite
from backend.trainer.state import TrainState, EvalState


class LossLogger(Callback):
    def __init__(self, output_path, plot_frequency):
        super().__init__()
        self.plot_frequency = plot_frequency
        self.output_path = Path(output_path) / "loss"
        os.makedirs(self.output_path, exist_ok=True)
        self.train_csv_path = self.output_path / "train_per_step.csv"
        self.eval_csv_path = self.output_path / "eval_per_step.csv"
        self.epoch_mean_csv_path = self.output_path / "losses_per_epoch.csv"

        self.train_epoch_history = []
        self.eval_epoch_history = []

        self.train_step_history = []
        self.eval_step_history = []

        self.lr_history = []

        self.train_loss_history = {}
        self.eval_loss_history = {}

    def on_train_batch_end(self, batch, logs: TrainState = None):
        self.train_epoch_history.append(logs.epoch)
        self.train_step_history.append(batch)
        self.lr_history.append(logs.optimizer.get_config()['learning_rate'])

        for loss_type, loss_value in logs.loss.items():
            if loss_type not in self.train_loss_history:
                self.train_loss_history[loss_type] = [loss_value]
            else:
                self.train_loss_history[loss_type].append(loss_value)

        if (batch + 1) % self.plot_frequency == 0:
            self.plot_loss_with_lr(
                self.train_loss_history,
                self.lr_history,
                self.train_epoch_history,
                'train_loss_per_step.png'
            )

        pd.DataFrame.from_dict({
            'epoch': self.train_epoch_history,
            'step': self.train_step_history,
            'lr': self.lr_history,
            **self.train_loss_history
        }).to_csv(
            self.train_csv_path, index=False,
            float_format=lambda x: round(x, 9)
        )

    def on_test_batch_end(self, batch, logs: EvalState = None):
        self.eval_epoch_history.append(logs.epoch)
        self.eval_step_history.append(batch)

        for loss_type, loss_value in logs.loss.items():
            if loss_type not in self.eval_loss_history:
                self.eval_loss_history[loss_type] = [loss_value]
            else:
                self.eval_loss_history[loss_type].append(loss_value)

        if (batch + 1) % self.plot_frequency == 0:
            self.plot_eval_loss(
                self.eval_loss_history,
                self.eval_epoch_history,
                'eval_loss_per_step.png'
            )

        pd.DataFrame.from_dict({
            'epoch': self.eval_epoch_history,
            'step': self.eval_step_history,
            **self.eval_loss_history
        }).to_csv(
            self.eval_csv_path, index=False,
            float_format=lambda x: round(x, 9)
        )

    def plot_eval_loss(self, loss_dict, epochs, name):
        steps = list(range(len(epochs)))
        rows = len(loss_dict) // 2 + len(loss_dict) % 2
        cols = 2
        for i, (label, losses) in enumerate(loss_dict.items(), start=1):
            plt.subplot(rows, cols, i)

            plt.plot(steps, losses, label=label)

            plt.title(label)
            plt.legend()

            x_labels = [str(x) for x in set(epochs)]
            x_ticks = [0] + [i for i in range(1, len(epochs)) if epochs[i] != epochs[i - 1]]
            plt.xticks(x_ticks, x_labels)

        plt.tight_layout()
        plt.savefig(self.output_path / name)
        plt.clf()

    def plot_loss_with_lr(self, loss_dict, lrs, epochs, name):
        plots = []
        for label, losses in loss_dict.items():
            fig = plt.figure()

            loss_plot = host_subplot(111, axes_class=AA.Axes)
            plt.subplots_adjust(right=0.75)
            learning_rate_plot = loss_plot.twinx()

            new_fixed_axis = learning_rate_plot.get_grid_helper().new_fixed_axis
            learning_rate_plot.axis['right'] = new_fixed_axis(
                loc='right',
                axes=learning_rate_plot
            )

            learning_rate_plot.axis['right'].toggle(all=True)

            loss_plot.set_xlabel('Epoch')
            loss_plot.set_ylabel('Loss')

            learning_rate_plot.set_ylabel('Learning rate')

            steps = list(range(len(lrs)))

            loss_plot.plot(steps, losses, label=label)

            learning_rate_plot.plot(steps, lrs, label='LR')
            loss_plot.legend()
            plt.title(label)

            x_labels = [str(x) for x in set(epochs)]
            x_ticks = [0] + [i for i in range(1, len(epochs)) if epochs[i] != epochs[i - 1]]
            plt.xticks(x_ticks, x_labels)

            plt.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plots.append(data)

            plt.clf()
            plt.close(fig)

        height, width, _ = plots[0].shape

        rows = len(plots) // 2 + len(plots) % 2
        cols = 2
        final_plot = np.ones((height*rows, width*cols, 3), dtype=int) * 255

        for i, plot in enumerate(plots):
            row = i // cols
            col = i % cols

            final_plot[row * height: (row+1) * height, col*width:(col+1) * width, :] = plot

        imwrite(str(self.output_path / name), final_plot)
