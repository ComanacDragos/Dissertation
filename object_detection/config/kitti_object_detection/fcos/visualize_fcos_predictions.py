from backend.visualizer.custom_visualizer import Visualizer
from backend.visualizer.object_detection_service import ObjectDetectionService
from backend.data_generator.inference_data_generator import InferenceDataGenerator
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig
from config.kitti_object_detection.fcos.common_fcos_config import FCOSCommonConfig
from config.object_detectors.fcos.fcos_model_config import FCOSModelConfig
from config.object_detectors.fcos.fcos_postprocessor_config import FCOSPostprocessingConfig


class VisualizeFCOSPredictionsConfig:
    # PATH_TO_WEIGHTS = FCOSCommonConfig.PREFIX / "train/checkpoints/model_13_0.5622483491897583_mAP.h5"
    # PATH_TO_WEIGHTS = FCOSCommonConfig.PREFIX / "train/checkpoints/model_17_0.5610086917877197_mAP.h5"
    PATH_TO_WEIGHTS = "outputs/kitti/fcos/arch/v5_3_scales_32_64_4_layers_64_filters_v2/train/checkpoints/model_18_0.5547028183937073_mAP.h5"

    @staticmethod
    def run():
        print(f"Loading {VisualizeFCOSPredictionsConfig.PATH_TO_WEIGHTS}")
        # model = tf.keras.models.load_model(VisualizeFCOSGTAndPredictionsConfig.PATH_TO_WEIGHTS)
        model = FCOSModelConfig.build(
            input_shape=KittiDataGeneratorConfig.INPUT_SHAPE,
            no_classes=len(KittiDataGeneratorConfig.LABELS),
            backbone_outputs=FCOSCommonConfig.BACKBONE_OUTPUTS,
            trainable_backbone=True,
            path_to_weights=VisualizeFCOSPredictionsConfig.PATH_TO_WEIGHTS
        )
        model.summary()
        postprocessor = FCOSPostprocessingConfig.build(
            image_size=KittiDataGeneratorConfig.INPUT_SHAPE[:2],
            max_boxes_per_image=KittiDataGeneratorConfig.MAX_BOXES_PER_IMAGE,
            batch_size=KittiDataGeneratorConfig.BATCH_SIZE
        )

        Visualizer(
            ObjectDetectionService(
                data_generator=InferenceDataGenerator(
                    KittiDataGeneratorConfig.ROOT,
                    KittiDataGeneratorConfig.CSV_PATH,
                    KittiDataGeneratorConfig.LABELS,
                    KittiDataGeneratorConfig.IMAGE_PREPROCESSOR
                ),
                model=lambda x: postprocessor(model(x))
            )
        )


if __name__ == '__main__':
    VisualizeFCOSPredictionsConfig.run()
