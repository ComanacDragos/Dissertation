from tensorflow.keras.callbacks import CallbackList
from backend.evaluation.map_metric import MeanAP
from backend.callbacks import *


class CallbacksConfig:
    PLOT_FREQUENCY = 1000

    @staticmethod
    def build(output_path, labels):
        return CallbackList([
            LossLogger(output_path, plot_frequency=CallbacksConfig.PLOT_FREQUENCY),
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
            ),
            ModelSaver(output_path, follow_metric='mAP'),
            CosineAnnealingScheduler(
                n_min=1e-6,
                n_max=5e-4,
                T=20,
            )
        ])
