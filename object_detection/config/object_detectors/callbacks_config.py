from tensorflow.keras.callbacks import CallbackList

from backend.callbacks import LossLogger, ModelSaver, BoxImagePlotter


class CallbacksConfig:
    PLOT_FREQUENCY = 1

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
        ])
