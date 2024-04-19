from tensorflow.keras.layers import Conv2DTranspose, Concatenate

from backend.model.blocks import ConvBlock
from backend.model.neck.fpn import FPNNeck
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

        return FPNNeck(upsample_block_generator)
