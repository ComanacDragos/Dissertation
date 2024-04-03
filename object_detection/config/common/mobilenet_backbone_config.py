import tensorflow as tf

from backend.model.generic_model import Sequential


class MobilenetBackboneConfig:
    @staticmethod
    def build(input_shape, alpha, trainable=False, outputs=('out_relu',)):
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False,
                                                                   alpha=alpha)
        if len(outputs) > 0:
            mobilenet = tf.keras.Model(inputs=mobilenet.input,
                                        outputs=[mobilenet.get_layer(name).output for name in
                                                 outputs],
                                        name="mobilenet")
        mobilenet.trainable = trainable
        return Sequential([preprocess, mobilenet])


if __name__ == '__main__':
    tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(370//2, 1220//2, 3), include_top=False,
                                                   alpha=0.5).summary()
