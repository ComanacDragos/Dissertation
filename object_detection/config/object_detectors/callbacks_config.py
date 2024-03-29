from tensorflow.keras.callbacks import CallbackList
from backend.evaluation.map_metric import MeanAP
from backend.callbacks import *


class CallbacksConfig:
    PLOT_FREQUENCY = 100

    @staticmethod
    def build(output_path, labels):
        return CallbackList([
            LossLogger(output_path, plot_frequency=CallbacksConfig.PLOT_FREQUENCY),
            ModelSaver(output_path),
            BoxImagePlotter(
                output_path,
                plot_frequency=CallbacksConfig.PLOT_FREQUENCY,
                labels=labels
            ),
            BoxEvaluation(
                output_path,
                labels=labels,
                metrics={
                    'mAP': MeanAP(labels, iou_threshold=0.5)
                }
            )
        ])
