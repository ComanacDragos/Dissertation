import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from overrides import overrides
from tensorflow.keras.callbacks import Callback

from backend.enums import DataType, ObjectDetectionOutputType, OutputType
from backend.logger import logger
from backend.trainer.state import EvalState
from backend.utils import to_json


class BoxEvaluation(Callback):
    def __init__(self, output_path, labels, metrics):
        super().__init__()
        self.id_to_class = {i: label for i, label in enumerate(labels)}
        self.output_path = Path(output_path) / "evaluation"
        os.makedirs(self.output_path, exist_ok=True)

        self.logs = None
        self.metrics = metrics
        self.history = {'epoch': []}

    @overrides
    def on_test_begin(self, logs: EvalState = None):
        self.logs = {
            "identifiers": [],
            "labels": [],
            "predictions": [],
        }

    @overrides
    def on_test_batch_end(self, batch, logs: EvalState = None):
        self.logs["identifiers"] += logs.inputs[DataType.IDENTIFIER]
        self.logs["labels"] += logs.inputs[DataType.LABEL]

        predictions = logs.predictions[ObjectDetectionOutputType.AFTER_FILTERING]

        per_image_predictions = [
            {
                OutputType.COORDINATES: coordinates,
                OutputType.CLASS_LABEL: labels,
                OutputType.CLASS_PROBABILITIES: probabilities
            } for coordinates, labels, probabilities in zip(
                predictions[OutputType.COORDINATES],
                predictions[OutputType.CLASS_LABEL],
                predictions[OutputType.CLASS_PROBABILITIES])
        ]

        self.logs["predictions"] += per_image_predictions

    @overrides
    def on_test_end(self, logs: EvalState = None):
        self.history['epoch'].append(logs.epoch)

        for metric_name, metric in self.metrics.items():
            logger.log(f"Computing {metric_name}...")
            result = metric(self.logs)
            if metric_name not in self.history:
                self.history[metric_name] = {
                    k: [v] for k, v in result.items()
                }
            else:
                for k, v in result.items():
                    self.history[metric_name][k].append(v)

            metric_history = self.history[metric_name]

            pd.DataFrame.from_dict({
                'epoch': self.history['epoch'],
                **metric_history
            }).to_csv(self.output_path / f"{metric_name}.csv", index=False)

            self._plot_history(metric_name, metric_history)
            self._plot_overview(logs.epoch, metric_name, result)

        os.makedirs(self.output_path / "dumps", exist_ok=True)
        to_json(self.logs, self.output_path / "dumps" / f"logs_{logs.epoch}.json")

    def _plot_overview(self, epoch, metric_name, results):
        output_path = self.output_path / "epoch_overview"
        os.makedirs(output_path, exist_ok=True)

        results = results.items()
        labels = [x[0] for x in results]
        values = [x[1] for x in results]

        plt.figure(figsize=(15, 5))
        sns.barplot({
            'labels': labels,
            'values': values
        }, y='labels', x='values', orient='h')
        plt.xticks([0.0] + list(np.linspace(0.1, 1, 10)))
        plt.savefig(output_path / f"epoch_{epoch}_{metric_name}_overview.png")
        plt.clf()
        plt.close()

    def _plot_history(self, metric_name, metric_history):
        cols = 2
        rows = len(metric_history) // 2
        if len(self.metrics) % cols > 0:
            rows += 1
        epochs = self.history['epoch']
        plt.figure(figsize=(10, 10))
        for i, (sub_metric_name, metric_history) in enumerate(metric_history.items(), start=1):
            plt.subplot(rows, cols, i)
            plt.title(sub_metric_name)
            plt.xlabel('epoch')

            not_none_values = [x for x in metric_history if x]

            if len(not_none_values) == 0:
                continue

            min_value, max_value = min(not_none_values), max(not_none_values)
            min_epoch, max_epoch = epochs[metric_history.index(min_value)], epochs[metric_history.index(max_value)]

            plt.plot(epochs, metric_history)

            plt.axhline(min_value, linestyle='--', color="blue")
            plt.axvline(min_epoch, linestyle='--', color="blue")

            plt.axhline(max_value, linestyle='--', color="red")
            plt.axvline(max_epoch, linestyle='--', color="red")

            ticks = [0.] + list(np.linspace(0.2, 1, 5))

            min_max_values = np.asarray([min_value, max_value])

            filtered_ticks = []
            for tick in ticks:
                if np.all(np.abs(min_max_values - tick) > 0.11):
                    filtered_ticks.append(tick)

            filtered_ticks = sorted(filtered_ticks + [min_value, max_value])

            plt.xticks(sorted([epochs[0], min_epoch, max_epoch, epochs[-1]]))
            plt.yticks(filtered_ticks)

        plt.tight_layout()
        plt.savefig(self.output_path / f"{metric_name}_history.png")
        plt.clf()
        plt.close()
