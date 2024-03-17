from backend.model.blocks import ConvBlock
from backend.model.generic_model import GenericBackbone
from config.common.layer_generators_config import *


class CustomResnetBackboneConfig:

    @staticmethod
    def build(conv_type):
        def block_generator(kernel_size=3, filters=32, strides=1, skip_connection_size=0):
            return ConvBlock(
                conv_generator(kernel_size, filters, strides),
                skip_connection_size=skip_connection_size,
                activation_generator=activation_generator,
                dropout_generator=dropout_generator,
                batch_norm_generator=batch_norm_generator,
            )

        return GenericBackbone(blocks=[
            block_generator(filters=512),
            block_generator(filters=256),
            block_generator(filters=256, skip_connection_size=2),
            block_generator(filters=128),
            block_generator(filters=128, skip_connection_size=2),
            block_generator(filters=64),
            block_generator(filters=64, skip_connection_size=2),
            block_generator(filters=32),
        ])
