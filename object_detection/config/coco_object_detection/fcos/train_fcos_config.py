import shutil
from pathlib import Path

from tensorflow.keras.optimizers import Adam

from backend.enums import Stage
from backend.trainer.generic_trainer import GenericTrainer
from backend.utils import set_seed
from config.coco_object_detection.data_generator_config import COCODataGeneratorConfig
from config.coco_object_detection.fcos.common_fcos_config import FCOSCommonConfig
from config.object_detectors.callbacks_config import CallbacksConfig
from config.object_detectors.fcos.fcos_loss_config import FCOSLossConfig
from config.object_detectors.fcos.fcos_model_config import FCOSModelConfig
from config.object_detectors.fcos.fcos_postprocessor_config import FCOSPostprocessingConfig
from config.object_detectors.fcos.fcos_preprocessor_config import FCOSPreprocessingConfig


class FCOSTrainerConfig:
    EXPERIMENT = FCOSCommonConfig.PREFIX / 'train'
    # EXPERIMENT = Path('outputs/test_train')

    EPOCHS = 20
    START_LR = 1e-4

    @staticmethod
    def build():
        return GenericTrainer(
            train_dataset=COCODataGeneratorConfig.build(Stage.TRAIN),
            val_dataset=COCODataGeneratorConfig.build(Stage.VAL),
            loss=FCOSLossConfig.build(
                image_size=COCODataGeneratorConfig.INPUT_SHAPE[:2],
                strides_weights=FCOSCommonConfig.STRIDES_WEIGHTS
            ),
            optimizer=Adam(learning_rate=FCOSTrainerConfig.START_LR, clipnorm=1.),
            callbacks=CallbacksConfig.build(FCOSTrainerConfig.EXPERIMENT, COCODataGeneratorConfig.LABELS),
            model=FCOSModelConfig.build(
                input_shape=COCODataGeneratorConfig.INPUT_SHAPE,
                no_classes=len(COCODataGeneratorConfig.LABELS),
                backbone_outputs=FCOSCommonConfig.BACKBONE_OUTPUTS,
            ),
            preprocessor=FCOSPreprocessingConfig.build(
                image_size=COCODataGeneratorConfig.INPUT_SHAPE[:2],
                no_classes=len(COCODataGeneratorConfig.LABELS),
                strides=FCOSCommonConfig.STRIDES,
                thresholds=FCOSCommonConfig.THRESHOLDS,
            ),
            postprocessor=FCOSPostprocessingConfig.build(
                image_size=COCODataGeneratorConfig.INPUT_SHAPE[:2],
                max_boxes_per_image=COCODataGeneratorConfig.MAX_BOXES_PER_IMAGE,
                batch_size=COCODataGeneratorConfig.BATCH_SIZE
            ),
            epochs=FCOSTrainerConfig.EPOCHS
        )


if __name__ == '__main__':
    set_seed(0)
    shutil.copytree('config', FCOSTrainerConfig.EXPERIMENT / 'config', dirs_exist_ok=True)
    FCOSTrainerConfig.build().train()
