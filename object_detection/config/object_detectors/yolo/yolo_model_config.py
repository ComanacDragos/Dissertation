from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from backend.logger import logger
from backend.model.generic_model import GenericModel
from backend.model.object_detection.yolo_model import YOLOHead
from config.common.layer_generators_config import conv_generator
from config.common.mobilenet_backbone_config import MobilenetBackboneConfig


class YOLOModelConfig:
    CONV_GENERATOR = conv_generator
    ALPHA = 1.0
    BACKBONE_OUTPUTS = ('block_9_add',)

    @staticmethod
    def build(input_shape, grid_size, no_anchors, no_classes, trainable_backbone=False, path_to_weights=None):
        inputs = Input(input_shape)
        x = GenericModel(
            backbone=MobilenetBackboneConfig.build(
                input_shape,
                YOLOModelConfig.ALPHA,
                outputs=YOLOModelConfig.BACKBONE_OUTPUTS
            ),
            head=YOLOHead(grid_size, no_anchors, no_classes, YOLOModelConfig.CONV_GENERATOR)
        )(inputs)
        model = Model(inputs=inputs, outputs=x, name='YOLODetector')
        if path_to_weights:
            model.load_weights(path_to_weights)
            if trainable_backbone:
                model.trainable = trainable_backbone
            logger.log(f"Loaded {path_to_weights}")
        model.summary(print_fn=logger)
        return model
