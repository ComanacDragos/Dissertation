import shutil
from pathlib import Path

import tensorflow.keras as K
from tensorflow.keras.optimizers import Adam

from backend.callbacks import LossLogger
from backend.enums import Stage
from backend.trainer.generic_trainer import GenericTrainer
from backend.utils import set_seed
from config.mnist_classification.data_generator_config import MNISTDataGeneratorConfig
from tensorflow.keras.callbacks import CallbackList
from backend.callbacks.classification_evaluation import ClassificationEvaluation
from backend.evaluation.accuracy_metric import Accuracy

def createModel(input_shape, num_classes):
    inputs = K.layers.Input(input_shape)
    x = K.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)
    x = K.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu')(x)
    x = K.layers.MaxPool2D()(x)
    x = K.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu')(x)
    x = K.layers.MaxPool2D()(x)
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(num_classes, activation='softmax')(x)

    model = K.models.Model(inputs=inputs, outputs=x)
    model.summary()
    return model


class MNISTTrainerConfig:
    EXPERIMENT = Path('outputs/mnist')
    # EXPERIMENT = Path('outputs/test_train')

    EPOCHS = 50
    START_LR = 1e-4

    @staticmethod
    def build():
        return GenericTrainer(
            train_dataset=MNISTDataGeneratorConfig.build(Stage.TRAIN),
            val_dataset=MNISTDataGeneratorConfig.build(Stage.VAL),
            loss=lambda *inputs: (K.losses.SparseCategoricalCrossentropy()(*inputs), {}),
            optimizer=Adam(learning_rate=MNISTTrainerConfig.START_LR),
            callbacks=CallbackList([
                ClassificationEvaluation(
                    output_path=MNISTTrainerConfig.EXPERIMENT,
                    labels=MNISTDataGeneratorConfig.LABELS,
                    metrics={
                        'acc': Accuracy(len(MNISTDataGeneratorConfig.LABELS))
                    }
                ),
                LossLogger(MNISTTrainerConfig.EXPERIMENT, plot_frequency=100)
            ]),
            model=createModel(
                input_shape=MNISTDataGeneratorConfig.INPUT_SHAPE,
                num_classes=len(MNISTDataGeneratorConfig.LABELS)
            ),
            preprocessor=None,
            postprocessor=None,
            epochs=MNISTTrainerConfig.EPOCHS
        )


if __name__ == '__main__':
    set_seed(0)
    shutil.copytree('config', MNISTTrainerConfig.EXPERIMENT / 'config', dirs_exist_ok=True)
    MNISTTrainerConfig.build().train()
