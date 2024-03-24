from abc import ABC

import tensorflow as tf
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils


class CustomLoss(tf.keras.losses.Loss, ABC):
    """
    Wrapper over the Loss class from tensorflow in order to return also components of the loss
    """
    def __call__(self, y_true, y_pred, sample_weight=None):
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight)
        with K.name_scope(self._name_scope), graph_ctx:
            ag_call = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
            losses, loss_dict = ag_call(y_true, y_pred)
            return losses_utils.compute_weighted_loss(
                losses, sample_weight, reduction=self._get_reduction()), loss_dict
