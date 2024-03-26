import albumentations as A

from backend.data_generator import KittiDataGenerator
from backend.enums import Stage, DataType


class KittiDataGeneratorConfig:
    ROOT = r"C:\Users\Dragos\datasets\KITTI"
    CSV_PATH = "csvs/kitti_toy.csv"
    BATCH_SIZE = 2
    CLASS_MAPPING = {
        "Pedestrian": "Pedestrian",
        "Truck": "Truck",
        "Car": "Car",
        "Cyclist": "Cyclist",
        # "DontCare": "DontCare",
        "Misc": "Misc",
        "Van": "Van",
        "Tram": "Tram",
        "Person_sitting": "Person_sitting"
    }
    LABELS = sorted(CLASS_MAPPING.values())
    SHUFFLE = False
    IMAGE_SHAPE = (370, 1220, 3)
    INPUT_SHAPE = (370 // 2, 1220 // 2, 3)
    MAX_BOXES_PER_IMAGE = 21

    @staticmethod
    def build(stage: Stage):
        return KittiDataGenerator(
            root=KittiDataGeneratorConfig.ROOT,
            csv_path=KittiDataGeneratorConfig.CSV_PATH,
            batch_size=KittiDataGeneratorConfig.BATCH_SIZE,
            stage=stage,
            class_mapping=KittiDataGeneratorConfig.CLASS_MAPPING,
            labels=KittiDataGeneratorConfig.LABELS,
            shuffle=KittiDataGeneratorConfig.SHUFFLE,
            augmentations=A.Compose([
                A.Crop(
                    x_max=KittiDataGeneratorConfig.IMAGE_SHAPE[1],  # width
                    y_max=KittiDataGeneratorConfig.IMAGE_SHAPE[0],  # height
                ),
                A.Resize(
                    width=KittiDataGeneratorConfig.INPUT_SHAPE[1],
                    height=KittiDataGeneratorConfig.INPUT_SHAPE[0]
                )
            ], bbox_params=A.BboxParams(format='pascal_voc', min_area=300, min_visibility=0.1,
                                        label_fields=['class_labels']))
        )


if __name__ == '__main__':
    print("Starting")
    KittiDataGeneratorConfig.BATCH_SIZE = 32
    db = KittiDataGeneratorConfig.build(Stage.VAL)
    print(len(db))
    data = db[0]
    print(f"Images shape {data[DataType.IMAGE].shape}")
