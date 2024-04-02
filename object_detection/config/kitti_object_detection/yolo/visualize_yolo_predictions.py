from pathlib import Path

import tensorflow as tf

from backend.visualizer.image_visualizer import VisTool
from backend.visualizer.inference_data_generator import InferenceDataGenerator
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig
from config.kitti_object_detection.yolo.common_yolo_config import YOLOCommonConfig
from config.object_detectors.yolo.yolo_postprocessor_config import YOLOPostprocessingConfig


class VisualizeYOLOPredictionsConfig:
    EXPERIMENT = Path("outputs/test_finetune")
    PATH_TO_WEIGHTS = "outputs/test_finetune/checkpoints/model_0_0.0_mAP.h5"

    ROOT = KittiDataGeneratorConfig.ROOT
    CSV = "csvs/kitti_toy.csv"

    @staticmethod
    def run():
        model = tf.keras.models.load_model(VisualizeYOLOPredictionsConfig.PATH_TO_WEIGHTS)
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
            InferenceDataGenerator(
                VisualizeYOLOPredictionsConfig.ROOT,
                VisualizeYOLOPredictionsConfig.CSV,
                KittiDataGeneratorConfig.LABELS,
                KittiDataGeneratorConfig.AUGMENTATIONS
            ),
            output=VisualizeYOLOPredictionsConfig.EXPERIMENT / "visualization",
            model=lambda x: postprocessor(model(x))
        ).run()


if __name__ == '__main__':
    VisualizeYOLOPredictionsConfig.run()
