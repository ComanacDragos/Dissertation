import shutil
from pathlib import Path

from tensorflow.keras.optimizers import Adam

from backend.enums import Stage
from backend.trainer.generic_trainer import GenericTrainer
from backend.utils import set_seed
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig
from config.object_detectors.callbacks_config import CallbacksConfig
from config.object_detectors.yolo.yolo_loss_config import YOLOLossConfig
from config.object_detectors.yolo.yolo_model_config import YOLOModelConfig
from config.object_detectors.yolo.yolo_postprocessor_config import YOLOPostprocessingConfig
from config.object_detectors.yolo.yolo_preprocessor_config import YOLOPreprocessingConfig
from config.kitti_object_detection.yolo.common_yolo_config import YOLOCommonConfig


class YOLOFinetuneConfig:
    EXPERIMENT = Path('outputs/test_finetune')

    EPOCHS = 2
    START_LR = 1e-5

    PATH_TO_WEIGHTS = "outputs/test_train/checkpoints/model_0_0.0_mAP.h5"
    KittiDataGeneratorConfig.BATCH_SIZE = 2

    @staticmethod
    def build():
        return GenericTrainer(
            train_dataset=KittiDataGeneratorConfig.build(Stage.TRAIN),
            val_dataset=KittiDataGeneratorConfig.build(Stage.VAL),
            loss=YOLOLossConfig.build(
                anchors=YOLOCommonConfig.ANCHORS,
                no_classes=len(KittiDataGeneratorConfig.LABELS),
                grid_size=YOLOCommonConfig.GRID_SIZE
            ),
            optimizer=Adam(learning_rate=YOLOFinetuneConfig.START_LR),
            callbacks=CallbacksConfig.build(YOLOFinetuneConfig.EXPERIMENT, KittiDataGeneratorConfig.LABELS),
            model=YOLOModelConfig.build(
                input_shape=KittiDataGeneratorConfig.INPUT_SHAPE,
                grid_size=YOLOCommonConfig.GRID_SIZE,
                no_anchors=len(YOLOCommonConfig.ANCHORS),
                no_classes=len(KittiDataGeneratorConfig.LABELS),
                trainable_backbone=True,
                path_to_weights=YOLOFinetuneConfig.PATH_TO_WEIGHTS
            ),
            preprocessor=YOLOPreprocessingConfig.build(
                image_size=KittiDataGeneratorConfig.INPUT_SHAPE[:2],
                grid_size=YOLOCommonConfig.GRID_SIZE,
                anchors=YOLOCommonConfig.ANCHORS,
                no_classes=len(KittiDataGeneratorConfig.LABELS),
                max_boxes_per_image=KittiDataGeneratorConfig.MAX_BOXES_PER_IMAGE
            ),
            postprocessor=YOLOPostprocessingConfig.build(
                image_size=KittiDataGeneratorConfig.INPUT_SHAPE[:2],
                grid_size=YOLOCommonConfig.GRID_SIZE,
                anchors=YOLOCommonConfig.ANCHORS,
                max_boxes_per_image=KittiDataGeneratorConfig.MAX_BOXES_PER_IMAGE,
                batch_size=KittiDataGeneratorConfig.BATCH_SIZE
            ),
            epochs=YOLOFinetuneConfig.EPOCHS
        )


if __name__ == '__main__':
    set_seed(0)
    shutil.copytree('config', YOLOFinetuneConfig.EXPERIMENT / 'config', dirs_exist_ok=True)
    YOLOFinetuneConfig.build().train()
