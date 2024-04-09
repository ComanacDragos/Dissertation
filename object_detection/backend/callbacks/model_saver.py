from tensorflow.keras.callbacks import Callback
import os
from backend.trainer.state import EvalState
from backend.logger import logger
import pandas as pd
from tensorflow.python.saved_model import save_options as save_options_lib


class ModelSaver(Callback):
    def __init__(self, output_path, follow_metric=None):
        super().__init__()
        self.output_path = output_path / "checkpoints"
        if follow_metric:
            self.metric_path = output_path / "evaluation" / f"{follow_metric}.csv"
        os.makedirs(self.output_path, exist_ok=True)
        self.follow_metric = follow_metric
        self.prev_output_path = None
        self.best_value = -1
        self.options = save_options_lib.SaveOptions()

    def on_test_end(self, logs: EvalState = None):
        if self.follow_metric:
            metric_value = list(pd.read_csv(self.metric_path).loc[:, "mAP"])[-1]
            if metric_value > self.best_value:
                self.best_value = metric_value

                output_path = self.output_path / f"model_{logs.epoch}_{metric_value}_{self.follow_metric}.h5"
                logger.log(f"Saving model {output_path}")
                logs.model.save(output_path, overwrite=True, options=self.options)
                if self.prev_output_path:
                    os.remove(self.prev_output_path)
                self.prev_output_path = output_path
            else:
                logger.log(f"Model did not improve from {self.best_value}, current value: {metric_value}")
