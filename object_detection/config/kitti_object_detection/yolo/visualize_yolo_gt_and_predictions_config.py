from pathlib import Path

import tensorflow as tf

from backend.enums import Stage
from backend.visualizer.image_visualizer import VisTool
from backend.visualizer.object_detection_vis_data_generator import ObjectDetectionVisDataGenerator
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig
from config.kitti_object_detection.yolo.common_yolo_config import YOLOCommonConfig
from config.object_detectors.yolo.yolo_postprocessor_config import YOLOPostprocessingConfig


class VisualizeYOLOGTAndPredictionsConfig:
    EXPERIMENT = Path("outputs/test_finetune")
    PATH_TO_WEIGHTS = "outputs/test_finetune/checkpoints/model_0_0.0_mAP.h5"

    @staticmethod
    def run():
        model = tf.keras.models.load_model(VisualizeYOLOGTAndPredictionsConfig.PATH_TO_WEIGHTS)
        model.summary()
        postprocessor = YOLOPostprocessingConfig.build(
            image_size=KittiDataGeneratorConfig.INPUT_SHAPE[:2],
            grid_size=YOLOCommonConfig.GRID_SIZE,
            anchors=YOLOCommonConfig.ANCHORS,
            max_boxes_per_image=KittiDataGeneratorConfig.MAX_BOXES_PER_IMAGE,
            batch_size=KittiDataGeneratorConfig.BATCH_SIZE
        )

        KittiDataGeneratorConfig.BATCH_SIZE = 1
        VisTool(
            ObjectDetectionVisDataGenerator(KittiDataGeneratorConfig.build(Stage.TRAIN)),
            output=VisualizeYOLOGTAndPredictionsConfig.EXPERIMENT / "visualization",
            model=lambda x: postprocessor(model(x))
        ).run()


if __name__ == '__main__':
    VisualizeYOLOGTAndPredictionsConfig.run()
