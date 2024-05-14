from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from backend.logger import logger
from backend.model.blocks import ConvBlock, InvertedResidualBlock
from backend.model.generic_model import MultiScaleModel, Sequential
from backend.model.object_detection.fcos_model import FCOSHead
from config.common.fpn_neck_config import FPNNeckConfig
from config.common.layer_generators_config import *
from config.common.mobilenet_backbone_config import MobilenetBackboneConfig


def conv_block_generator(filters, strides=1, add_skip_connection=False):
    return ConvBlock(
        conv_generator(3, filters, strides),
        activation_generator=activation_generator,
        batch_norm_generator=batch_norm_generator,
        add_skip_connection=add_skip_connection,
        dropout_generator=dropout_generator,
    )


def inverted_block_generator(expand_filters, squeeze_filters):
    return InvertedResidualBlock(
        expand_conv_generator=conv_generator(1, expand_filters),
        squeeze_conv_generator=conv_generator(1, squeeze_filters),
        depthwise_conv_generator=depthwise_conv_generator(3),
        activation_generator=activation_generator,
        batch_norm_generator=batch_norm_generator,
    )


class FCOSModelConfig:
    CONV_GENERATOR = conv_generator
    ALPHA = 1.

    @staticmethod
    def build(input_shape, no_classes, backbone_outputs=None, trainable_backbone=False, path_to_weights=None):
        inputs = Input(input_shape)
        x = MultiScaleModel(
            backbone=MobilenetBackboneConfig.build(
                input_shape,
                FCOSModelConfig.ALPHA,
                outputs=backbone_outputs
            ),
            neck=FPNNeckConfig.build(),
            head=FCOSHead(
                no_classes=no_classes,
                conv_generator=FCOSModelConfig.CONV_GENERATOR,
                classification_branch=Sequential([
                    conv_block_generator(64),
                    conv_block_generator(64),
                    conv_block_generator(64),
                    conv_block_generator(64),
                ]),
                regression_branch=Sequential([
                    conv_block_generator(64),
                    conv_block_generator(64),
                    conv_block_generator(64),
                    conv_block_generator(64),
                ])
            )
        )(inputs)
        model = Model(inputs=inputs, outputs=x, name='FCOSDetector')
        if path_to_weights:
            model.load_weights(path_to_weights)
            if trainable_backbone:
                model.trainable = trainable_backbone
            logger.log(f"Loaded {path_to_weights}")
        model.summary(print_fn=logger)
        return model
