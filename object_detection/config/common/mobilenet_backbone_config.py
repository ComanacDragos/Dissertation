import tensorflow as tf

from backend.model.generic_model import GenericBackbone


class MobilenetBackboneConfig:
    @staticmethod
    def build(input_shape, alpha):
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False,
                                                                   alpha=alpha)
        return GenericBackbone([preprocess, mobilenet])
