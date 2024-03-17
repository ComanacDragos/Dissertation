import albumentations as A

from backend.data_generator import KittiDataGenerator
from backend.enums import Stage, DataType


class KittiDataGeneratorConfig:
    ROOT = r"C:\Users\Dragos\datasets\KITTI"
    CSV_PATH = "csvs/kitti.csv"
    BATCH_SIZE = 8
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
    SHUFFLE = False
    IMAGE_SHAPE = (370, 1220, 3)

    @staticmethod
    def build(stage: Stage):
        return KittiDataGenerator(
            root=KittiDataGeneratorConfig.ROOT,
            csv_path=KittiDataGeneratorConfig.CSV_PATH,
            batch_size=KittiDataGeneratorConfig.BATCH_SIZE,
            stage=stage,
            class_mapping=KittiDataGeneratorConfig.CLASS_MAPPING,
            shuffle=KittiDataGeneratorConfig.SHUFFLE,
            augmentations=A.Compose([
                A.Crop(
                    x_max=KittiDataGeneratorConfig.IMAGE_SHAPE[1],  # width
                    y_max=KittiDataGeneratorConfig.IMAGE_SHAPE[0],  # height
                )
            ], bbox_params=A.BboxParams(format='pascal_voc', min_area=400, min_visibility=0.1,
                                        label_fields=['class_labels']))
        )


if __name__ == '__main__':
    print("Starting")
    KittiDataGeneratorConfig.BATCH_SIZE = 32
    db = KittiDataGeneratorConfig.build(Stage.VAL)
    print(len(db))
    data = db[0]
    print(f"Images shape {data[DataType.IMAGE].shape}")
