from tensorflow.keras.layers import Conv2DTranspose, Concatenate

from backend.model.blocks import ConvBlock, InvertedResidualBlock
from config.common.layer_generators_config import *


class UnetNeckConfig:

    @staticmethod
    def build():
        def inverted_block_generator(expand_filters, squeeze_filters):
            return InvertedResidualBlock(
                expand_conv_generator=conv_generator(1, expand_filters),
                squeeze_conv_generator=conv_generator(1, squeeze_filters),
                depthwise_conv_generator=depthwise_conv_generator(3),
                activation_generator=activation_generator,
                batch_norm_generator=batch_norm_generator,
            )

        def conv_block_generator(filters, strides=1):
            return ConvBlock(
                conv_generator(3, filters, strides),
                activation_generator=activation_generator,
                batch_norm_generator=batch_norm_generator,
                # dropout_generator=dropout_generator,
            )

        def upsample_block_generator(filters, strides):
            return ConvBlock(
                conv_generator(3, filters, strides, conv_type=Conv2DTranspose),
                activation_generator=activation_generator,
                batch_norm_generator=batch_norm_generator,
            )

        def unet_neck(inputs):
            x = inputs[-1]
            for layer in reversed(inputs[:-1]):
                x = upsample_block_generator(filters=layer.shape[-1], strides=int(layer.shape[1] / x.shape[1]))(x)
                x = Concatenate()([x, layer])

            x = conv_block_generator(filters=128)(x)

            x = inverted_block_generator(512, 128)(x)
            x = inverted_block_generator(512, 128)(x)

            # x = conv_block_generator(filters=64)(x)
            #
            # x = inverted_block_generator(256, 64)(x)
            # x = inverted_block_generator(256, 64)(x)
            #
            # x = conv_block_generator(filters=32)(x)
            #
            # x = inverted_block_generator(128, 32)(x)
            # x = inverted_block_generator(128, 32)(x)

            return x

        return unet_neck
