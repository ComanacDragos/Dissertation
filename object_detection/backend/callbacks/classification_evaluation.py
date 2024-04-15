import tensorflow as tf
from overrides import overrides

from backend.callbacks.box_evaluation import BoxEvaluation
from backend.enums import DataType
from backend.trainer.state import EvalState


class ClassificationEvaluation(BoxEvaluation):
    @overrides
    def on_test_batch_end(self, batch, logs: EvalState = None):
        self.logs["identifiers"] += logs.inputs[DataType.IDENTIFIER].tolist()
        self.logs["labels"] += logs.inputs[DataType.LABEL].numpy().tolist()

        predictions = logs.predictions
        predictions = tf.argmax(predictions, axis=-1)

        self.logs["predictions"] += predictions.numpy().tolist()
