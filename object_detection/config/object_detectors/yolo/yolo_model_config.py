from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from backend.model.generic_model import GenericModel
from backend.model.object_detection.yolo_model import YOLOHead
from backend.utils import logger
from config.common.layer_generators_config import conv_generator
from config.common.mobilenet_backbone_config import MobilenetBackboneConfig


class YOLOModelConfig:
    CONV_GENERATOR = conv_generator
    ALPHA = 1.0
    TRAIN_BACKBONE = False

    @staticmethod
    def build(input_shape, grid_size, no_anchors, no_classes):
        inputs = Input(input_shape)
        x = GenericModel(
            backbone=MobilenetBackboneConfig.build(
                input_shape,
                YOLOModelConfig.ALPHA,
                trainable=YOLOModelConfig.TRAIN_BACKBONE
            ),
            head=YOLOHead(grid_size, no_anchors, no_classes, YOLOModelConfig.CONV_GENERATOR)
        )(inputs)
        model = Model(inputs=inputs, outputs=x, name='YOLODetector')
        model.summary(print_fn=logger)
        return model
