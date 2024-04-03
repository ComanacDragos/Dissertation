from backend.model.blocks import ConvBlock
from backend.model.generic_model import Sequential
from config.common.layer_generators_config import *


class CustomResnetBackboneConfig:

    @staticmethod
    def build(conv_type=Conv2D):
        def block_generator(kernel_size=3, filters=32, strides=1, skip_connection_size=0):
            return ConvBlock(
                conv_generator(kernel_size, filters, strides, conv_type=conv_type),
                skip_connection_size=skip_connection_size,
                activation_generator=activation_generator,
                # dropout_generator=dropout_generator,
                batch_norm_generator=batch_norm_generator,
            )

        return Sequential(blocks=[
            # block_generator(filters=256, strides=2),
            # block_generator(filters=256, skip_connection_size=2),
            # block_generator(filters=128, strides=2),
            # block_generator(filters=128, skip_connection_size=2),
            block_generator(filters=64, strides=2),
            block_generator(filters=64, skip_connection_size=2),
            block_generator(filters=32, strides=2),
            block_generator(filters=32, skip_connection_size=2),
            block_generator(filters=16, strides=2),
            block_generator(filters=16, skip_connection_size=2),
            block_generator(filters=8, strides=2),
            block_generator(filters=8, skip_connection_size=2),
        ])
