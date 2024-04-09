from backend.model.blocks import ConvBlock
from backend.model.generic_model import Sequential
from config.common.layer_generators_config import *


class SimpleNeckConfig:

    @staticmethod
    def build(conv_type=Conv2D):
        def block_generator(kernel_size=3, filters=32, strides=1, skip_connection_size=0):
            return ConvBlock(
                conv_generator(kernel_size, filters, strides, conv_type=conv_type),
                skip_connection_size=skip_connection_size,
                activation_generator=activation_generator,
                batch_norm_generator=batch_norm_generator,
            )

        return Sequential(blocks=[
            # block_generator(filters=128),
            block_generator(filters=64),
            # block_generator(filters=32),
        ])
