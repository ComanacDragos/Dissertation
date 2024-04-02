from pathlib import Path

from backend.enums import Stage
from backend.visualizer.image_visualizer import VisTool
from backend.visualizer.object_detection_vis_data_generator import ObjectDetectionVisDataGenerator
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig


class VisualizeGTConfig:
    EXPERIMENT = Path("outputs/kitti")

    @staticmethod
    def run():
        KittiDataGeneratorConfig.BATCH_SIZE = 1
        VisTool(
            ObjectDetectionVisDataGenerator(KittiDataGeneratorConfig.build(Stage.TRAIN)),
            output=VisualizeGTConfig.EXPERIMENT / "visualization"
        ).run()


if __name__ == '__main__':
    VisualizeGTConfig.run()
