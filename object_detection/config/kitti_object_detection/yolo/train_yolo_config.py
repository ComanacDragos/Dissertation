import shutil
from pathlib import Path

from tensorflow.keras.optimizers import Adam

from backend.enums import Stage
from backend.trainer.generic_trainer import GenericTrainer
from backend.utils import set_seed
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig
from config.kitti_object_detection.yolo.common_yolo_config import YOLOCommonConfig
from config.object_detectors.callbacks_config import CallbacksConfig
from config.object_detectors.yolo.yolo_loss_config import YOLOLossConfig
from config.object_detectors.yolo.yolo_model_config import YOLOModelConfig
from config.object_detectors.yolo.yolo_postprocessor_config import YOLOPostprocessingConfig
from config.object_detectors.yolo.yolo_preprocessor_config import YOLOPreprocessingConfig


class YOLOTrainerConfig:
    EXPERIMENT = Path('outputs/yolo_train') # _v7_lr_t=10_dropout=50%

    EPOCHS = 10
    START_LR = 1e-4

    @staticmethod
    def build():
        return GenericTrainer(
            train_dataset=KittiDataGeneratorConfig.build(Stage.TRAIN),
            val_dataset=KittiDataGeneratorConfig.build(Stage.VAL),
            loss=YOLOLossConfig.build(
                anchors=YOLOCommonConfig.ANCHORS,
                no_classes=len(KittiDataGeneratorConfig.LABELS),
                grid_size=YOLOCommonConfig.GRID_SIZE,
                max_height=KittiDataGeneratorConfig.INPUT_SHAPE[0],
                max_width=KittiDataGeneratorConfig.INPUT_SHAPE[1]
            ),
            optimizer=Adam(learning_rate=YOLOTrainerConfig.START_LR, epsilon=1e-8, decay=0.0, clipnorm=1.0),
            callbacks=CallbacksConfig.build(YOLOTrainerConfig.EXPERIMENT, KittiDataGeneratorConfig.LABELS),
            model=YOLOModelConfig.build(
                input_shape=KittiDataGeneratorConfig.INPUT_SHAPE,
                grid_size=YOLOCommonConfig.GRID_SIZE,
                no_anchors=len(YOLOCommonConfig.ANCHORS),
                no_classes=len(KittiDataGeneratorConfig.LABELS),
                backbone_outputs=YOLOCommonConfig.BACKBONE_OUTPUTS,
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
            epochs=YOLOTrainerConfig.EPOCHS
        )


if __name__ == '__main__':
    # set_seed(0)
    shutil.copytree('config', YOLOTrainerConfig.EXPERIMENT / 'config', dirs_exist_ok=True)
    YOLOTrainerConfig.build().train()
