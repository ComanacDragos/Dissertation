from pathlib import Path

from overrides import overrides
from tensorflow.keras.callbacks import Callback

from backend.enums import DataType, ObjectDetectionOutputType, OutputType, LabelType
from backend.trainer.state import EvalState


class BoxEvaluation(Callback):
    def __init__(self, output_path, labels, metrics):
        super().__init__()
        self.id_to_class = {i: label for i, label in enumerate(labels)}
        self.output_path = Path(output_path) / "evaluation"

        self.logs = None
        self.metrics = metrics
        self.history = {'epoch': []}

    @overrides
    def on_test_begin(self, logs: EvalState = None):
        self.logs = {
            "identifiers": [],
            "labels": [],
            "predictions": []
        }

    @overrides
    def on_test_batch_end(self, batch, logs: EvalState = None):
        self.logs["identifiers"] += logs.inputs[DataType.IDENTIFIER]
        self.logs["labels"] += logs.inputs[DataType.LABEL]

        predictions = logs.predictions[ObjectDetectionOutputType.BEFORE_FILTERING]

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
            result = metric(self.logs)
            if metric_name in self.history:
                self.history[metric_name].append(result)
            else:
                self.history[metric_name] = [result]
