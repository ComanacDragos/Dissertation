from backend.enums import Stage
from backend.visualizer.custom_visualizer import Visualizer
from backend.visualizer.object_detection_service import ObjectDetectionService
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig


class VisualizeGTConfig:
    @staticmethod
    def run():
        KittiDataGeneratorConfig.BATCH_SIZE = 1
        Visualizer(
            ObjectDetectionService(
                KittiDataGeneratorConfig.build(Stage.ALL)
            )
        )


if __name__ == '__main__':
    VisualizeGTConfig.run()
