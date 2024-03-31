import shutil
from pathlib import Path

from tensorflow.keras.optimizers import Adam

from backend.enums import Stage
from backend.trainer.generic_trainer import GenericTrainer
from backend.utils import set_seed, open_json
from config.object_detectors.callbacks_config import CallbacksConfig
from config.object_detectors.yolo.yolo_loss_config import YOLOLossConfig
from config.object_detectors.yolo.yolo_model_config import YOLOModelConfig
from config.object_detectors.yolo.yolo_postprocessor_config import YOLOPostprocessingConfig
from config.object_detectors.yolo.yolo_preprocessor_config import YOLOPreprocessingConfig
from data_generator_config import KittiDataGeneratorConfig


class YOLOTrainerConfig:
    EXPERIMENT = Path('outputs/test_train')

    EPOCHS = 100
    START_LR = 1e-4

    ANCHORS = open_json("scripts/kitti_anchors.json")["3"]
    GRID_SIZE = (6, 20)

    @staticmethod
    def build():
        input_shape = KittiDataGeneratorConfig.INPUT_SHAPE
        no_classes = len(set(KittiDataGeneratorConfig.CLASS_MAPPING.keys()))
        return GenericTrainer(
            train_dataset=KittiDataGeneratorConfig.build(Stage.TRAIN),
            val_dataset=KittiDataGeneratorConfig.build(Stage.VAL),
            loss=YOLOLossConfig.build(
                anchors=YOLOTrainerConfig.ANCHORS,
                no_classes=no_classes,
                grid_size=YOLOTrainerConfig.GRID_SIZE
            ),
            optimizer=Adam(learning_rate=YOLOTrainerConfig.START_LR),
            callbacks=CallbacksConfig.build(YOLOTrainerConfig.EXPERIMENT, KittiDataGeneratorConfig.LABELS),
            model=YOLOModelConfig.build(
                input_shape=input_shape,
                grid_size=YOLOTrainerConfig.GRID_SIZE,
                no_anchors=len(YOLOTrainerConfig.ANCHORS),
                no_classes=no_classes
            ),
            preprocessor=YOLOPreprocessingConfig.build(
                image_size=input_shape[:2],
                grid_size=YOLOTrainerConfig.GRID_SIZE,
                anchors=YOLOTrainerConfig.ANCHORS,
                no_classes=no_classes,
                max_boxes_per_image=KittiDataGeneratorConfig.MAX_BOXES_PER_IMAGE
            ),
            postprocessor=YOLOPostprocessingConfig.build(
                image_size=input_shape[:2],
                grid_size=YOLOTrainerConfig.GRID_SIZE,
                anchors=YOLOTrainerConfig.ANCHORS,
                max_boxes_per_image=KittiDataGeneratorConfig.MAX_BOXES_PER_IMAGE,
                batch_size=KittiDataGeneratorConfig.BATCH_SIZE
            ),
            epochs=YOLOTrainerConfig.EPOCHS
        )


if __name__ == '__main__':
    set_seed(0)
    shutil.copytree('config', YOLOTrainerConfig.EXPERIMENT / 'config', dirs_exist_ok=True)
    YOLOTrainerConfig.build().train()
