from pathlib import Path

from backend.enums import Stage
from backend.visualizer.image_visualizer import VisTool
from backend.visualizer.object_detection_vis_data_generator import ObjectDetectionVisDataGenerator
from config.public_transport_object_detection.data_generator_config import PublicTransportDataGeneratorConfig


class VisualizeGTConfig:
    EXPERIMENT = Path("outputs/kitti")

    @staticmethod
    def run():
        PublicTransportDataGeneratorConfig.BATCH_SIZE = 1
        VisTool(
            ObjectDetectionVisDataGenerator(PublicTransportDataGeneratorConfig.build(Stage.ALL)),
            output=VisualizeGTConfig.EXPERIMENT / "visualization"
        ).run()


if __name__ == '__main__':
    VisualizeGTConfig.run()
