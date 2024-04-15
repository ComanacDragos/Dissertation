from backend.data_generator import MNISTDataGenerator
from backend.enums import Stage


class MNISTDataGeneratorConfig:
    BATCH_SIZE = 512
    LABELS = [str(x) for x in range(10)]
    IMAGE_SHAPE = (28, 28, 1)
    INPUT_SHAPE = (28, 28, 1)

    @staticmethod
    def build(stage: Stage):
        return MNISTDataGenerator(
            batch_size=MNISTDataGeneratorConfig.BATCH_SIZE,
            stage=stage,
        )
