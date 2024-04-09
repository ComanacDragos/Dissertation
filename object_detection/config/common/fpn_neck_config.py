from tensorflow.keras.layers import Conv2DTranspose, Concatenate

from backend.model.blocks import ConvBlock
from config.common.layer_generators_config import *


class FPNNeckConfig:

    @staticmethod
    def build():
        def upsample_block_generator(filters, strides):
            return ConvBlock(
                conv_generator(3, filters, strides, conv_type=Conv2DTranspose),
                activation_generator=activation_generator,
                batch_norm_generator=batch_norm_generator,
            )

        def fpn_neck(inputs):
            x = inputs[-1]
            outputs = [x]
            for layer in reversed(inputs[:-1]):
                x = upsample_block_generator(filters=layer.shape[-1], strides=int(layer.shape[1] / x.shape[1]))(x)
                x = Concatenate()([x, layer])
                outputs.append(x)
            return outputs

        return fpn_neck
