from tensorflow.keras.callbacks import CallbackList

from backend.callbacks import *


class CallbacksConfig:
    PLOT_FREQUENCY = 10

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
                labels=labels
            )
        ])