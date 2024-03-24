from tensorflow.keras.callbacks import Callback
import os
from backend.trainer.state import EvalState


class ModelSaver(Callback):
    def __init__(self, output_path, follow_metric=None):
        super().__init__()
        self.output_path = output_path / "checkpoints"
        os.makedirs(self.output_path, exist_ok=True)
        self.follow_metric = follow_metric

    def on_test_end(self, logs: EvalState = None):
        logs.model.save(self.output_path / f"model_{logs.epoch}.h5", overwrite=True)
