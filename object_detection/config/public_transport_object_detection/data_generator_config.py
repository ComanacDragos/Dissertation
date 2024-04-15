from backend.data_generator import PublicTransportDataGenerator
from backend.enums import Stage, DataType


class PublicTransportDataGeneratorConfig:
    ROOT = r"C:\Users\Dragos\datasets\OID\OID\PublicTransportFilteredProcessed"
    CSV_PATH = "csvs/PublicTransportFilteredProcessed.csv"
    BATCH_SIZE = 8
    CLASS_MAPPING = {
        "Bus": "bus",
        "Car": "car",
        "Vehicle registration plate": "plate"
    }
    LABELS = sorted(set(CLASS_MAPPING.values()))
    IMAGE_SHAPE = (416, 416, 3)
    INPUT_SHAPE = IMAGE_SHAPE
    MAX_BOXES_PER_IMAGE = 111

    AUGMENTATIONS = None

    @staticmethod
    def build(stage: Stage):
        return PublicTransportDataGenerator(
            root=PublicTransportDataGeneratorConfig.ROOT,
            csv_path=PublicTransportDataGeneratorConfig.CSV_PATH,
            batch_size=PublicTransportDataGeneratorConfig.BATCH_SIZE,
            stage=stage,
            class_mapping=PublicTransportDataGeneratorConfig.CLASS_MAPPING,
            labels=PublicTransportDataGeneratorConfig.LABELS,
            augmentations=PublicTransportDataGeneratorConfig.AUGMENTATIONS
        )


if __name__ == '__main__':
    print("Starting")
    PublicTransportDataGeneratorConfig.BATCH_SIZE = 32
    db = PublicTransportDataGeneratorConfig.build(Stage.VAL)
    print(len(db))
    data = db[0]
    print(f"Images shape {data[DataType.IMAGE].shape}")
