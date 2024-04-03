from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from backend.logger import logger
from backend.model.generic_model import GenericModel
from backend.model.object_detection.yolo_model import YOLOHead
from config.common.layer_generators_config import conv_generator
from config.common.mobilenet_backbone_config import MobilenetBackboneConfig
from config.common.custom_resnet_backbone_config import CustomResnetBackboneConfig
from config.common.simple_neck_config import SimpleNeckConfig


class YOLOModelConfig:
    CONV_GENERATOR = conv_generator
    ALPHA = 1.0

    @staticmethod
    def build(input_shape, grid_size, no_anchors, no_classes, backbone_outputs=None, trainable_backbone=False, path_to_weights=None):
        inputs = Input(input_shape)
        x = GenericModel(
            backbone=MobilenetBackboneConfig.build(
                input_shape,
                YOLOModelConfig.ALPHA,
                outputs=backbone_outputs
            ),
            # backbone=CustomResnetBackboneConfig.build(),
            # neck=SimpleNeckConfig.build(),
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
