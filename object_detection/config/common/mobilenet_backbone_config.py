import tensorflow as tf

from backend.model.generic_model import GenericBackbone


class MobilenetBackboneConfig:
    @staticmethod
    def build(input_shape, alpha, trainable=False):
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False,
                                                                   alpha=alpha)
        mobilenet.trainable = trainable
        return GenericBackbone([preprocess, mobilenet])


if __name__ == '__main__':
    tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(370//2, 1220//2, 3), include_top=False,
                                                   alpha=0.5).summary()
