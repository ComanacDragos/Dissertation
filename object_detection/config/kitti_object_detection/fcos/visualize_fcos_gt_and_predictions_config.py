from backend.enums import Stage
from backend.visualizer.custom_visualizer import Visualizer
from backend.visualizer.object_detection_service import ObjectDetectionService
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig
from config.kitti_object_detection.fcos.common_fcos_config import FCOSCommonConfig
from config.object_detectors.fcos.fcos_model_config import FCOSModelConfig
from config.object_detectors.fcos.fcos_postprocessor_config import FCOSPostprocessingConfig


class VisualizeFCOSGTAndPredictionsConfig:
    PATH_TO_WEIGHTS = "models/kitti_reduced_model_17_0.5654428601264954_mAP.h5"

    @staticmethod
    def run():
        print(f"Loading {VisualizeFCOSGTAndPredictionsConfig.PATH_TO_WEIGHTS}")
        # model = tf.keras.models.load_model(VisualizeFCOSGTAndPredictionsConfig.PATH_TO_WEIGHTS)
        model = FCOSModelConfig.build(
            input_shape=KittiDataGeneratorConfig.INPUT_SHAPE,
            no_classes=len(KittiDataGeneratorConfig.LABELS),
            backbone_outputs=FCOSCommonConfig.BACKBONE_OUTPUTS,
            trainable_backbone=True,
            path_to_weights=VisualizeFCOSGTAndPredictionsConfig.PATH_TO_WEIGHTS
        )
        model.summary()
        postprocessor = FCOSPostprocessingConfig.build(
            image_size=KittiDataGeneratorConfig.INPUT_SHAPE[:2],
            max_boxes_per_image=KittiDataGeneratorConfig.MAX_BOXES_PER_IMAGE,
            batch_size=KittiDataGeneratorConfig.BATCH_SIZE
        )

        KittiDataGeneratorConfig.BATCH_SIZE = 1
        KittiDataGeneratorConfig.AUGMENTATIONS = None
        KittiDataGeneratorConfig.SHUFFLE = False
        Visualizer(
            ObjectDetectionService(
                data_generator=KittiDataGeneratorConfig.build(Stage.VAL),
                model=lambda x: postprocessor(model(x))
            )
        )


if __name__ == '__main__':
    VisualizeFCOSGTAndPredictionsConfig.run()
