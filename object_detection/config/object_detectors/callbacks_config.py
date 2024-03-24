from tensorflow.keras.callbacks import CallbackList
from backend.callbacks import LossLogger


class CallbacksConfig:
    @staticmethod
    def build(output_path):
        return CallbackList([
            LossLogger(output_path, 'train.csv'),
        ])
