import tensorflow as tf

from backend.model.generic_model import GenericBackbone


class Resnet50BackboneConfig:
    @staticmethod
    def build(input_shape):
        preprocess = tf.keras.applications.resnet50.preprocess_input
        resnet = tf.keras.applications.resnet50.ResNet50(input_shape=input_shape, include_top=False)
        return GenericBackbone([preprocess, resnet])
