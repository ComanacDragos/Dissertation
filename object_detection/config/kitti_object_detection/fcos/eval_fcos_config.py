import shutil
from pathlib import Path

from backend.enums import Stage
from backend.trainer.generic_trainer import GenericTrainer
from backend.utils import set_seed
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig
from config.kitti_object_detection.fcos.common_fcos_config import FCOSCommonConfig
from config.object_detectors.callbacks_config import CallbacksConfig
from config.object_detectors.fcos.fcos_loss_config import FCOSLossConfig
from config.object_detectors.fcos.fcos_model_config import FCOSModelConfig
from config.object_detectors.fcos.fcos_postprocessor_config import FCOSPostprocessingConfig
from config.object_detectors.fcos.fcos_preprocessor_config import FCOSPreprocessingConfig


class FCOSEvalConfig:
    EXPERIMENT = Path(f'{FCOSCommonConfig.PREFIX}/eval_min_score_02_nms_02_iou_01')
    PATH_TO_WEIGHTS = f"{FCOSCommonConfig.PREFIX}/train/checkpoints/model_17_0.1852328479290008_mAP.h5"

    @staticmethod
    def build():
        return GenericTrainer(
            train_dataset=KittiDataGeneratorConfig.build(Stage.TRAIN),
            val_dataset=KittiDataGeneratorConfig.build(Stage.VAL),
            loss=FCOSLossConfig.build(
                image_size=KittiDataGeneratorConfig.INPUT_SHAPE[:2],
                strides_weights=FCOSCommonConfig.STRIDES_WEIGHTS
            ),
            optimizer=None,
            callbacks=CallbacksConfig.build(FCOSEvalConfig.EXPERIMENT, KittiDataGeneratorConfig.LABELS),
            model=FCOSModelConfig.build(
                input_shape=KittiDataGeneratorConfig.INPUT_SHAPE,
                no_classes=len(KittiDataGeneratorConfig.LABELS),
                backbone_outputs=FCOSCommonConfig.BACKBONE_OUTPUTS,
                trainable_backbone=True,
                path_to_weights=FCOSEvalConfig.PATH_TO_WEIGHTS
            ),
            preprocessor=FCOSPreprocessingConfig.build(
                image_size=KittiDataGeneratorConfig.INPUT_SHAPE[:2],
                no_classes=len(KittiDataGeneratorConfig.LABELS),
                strides=FCOSCommonConfig.STRIDES,
                thresholds=FCOSCommonConfig.THRESHOLDS,
            ),
            postprocessor=FCOSPostprocessingConfig.build(
                image_size=KittiDataGeneratorConfig.INPUT_SHAPE[:2],
                max_boxes_per_image=KittiDataGeneratorConfig.MAX_BOXES_PER_IMAGE,
                batch_size=KittiDataGeneratorConfig.BATCH_SIZE
            ),
            epochs=0
        )


if __name__ == '__main__':
    set_seed(0)
    shutil.copytree('config', FCOSEvalConfig.EXPERIMENT / 'config', dirs_exist_ok=True)
    FCOSEvalConfig.build().eval_loop(0)
