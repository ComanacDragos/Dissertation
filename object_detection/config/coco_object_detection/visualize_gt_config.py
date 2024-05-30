from backend.enums import Stage
from backend.visualizer.custom_visualizer import Visualizer
from backend.visualizer.object_detection_service import ObjectDetectionService
from config.coco_object_detection.data_generator_config import COCODataGeneratorConfig


class VisualizeGTConfig:
    @staticmethod
    def run():
        COCODataGeneratorConfig.BATCH_SIZE = 1
        Visualizer(
            ObjectDetectionService(
                COCODataGeneratorConfig.build(Stage.ALL)
            )
        )


if __name__ == '__main__':
    VisualizeGTConfig.run()
