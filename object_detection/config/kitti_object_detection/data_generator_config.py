from backend.data_generator import KittiDataGenerator
from backend.enums import Stage


class KittiDataGeneratorConfig:
    ROOT = r"C:\Users\Dragos\datasets\KITTI"
    CSV_PATH = "csvs/kitti.csv"
    BATCH_SIZE = 8
    CLASS_MAPPING = {
        "Pedestrian": "Pedestrian",
        "Truck": "Truck",
        "Car": "Car",
        "Cyclist": "Cyclist",
        "DontCare": "DontCare",
        "Misc": "Misc",
        "Van": "Van",
        "Tram": "Tram",
        "Person_sitting": "Person_sitting"
    }

    @staticmethod
    def build(stage: Stage):
        return KittiDataGenerator(
            root=KittiDataGeneratorConfig.ROOT,
            csv_path=KittiDataGeneratorConfig.CSV_PATH,
            batch_size=KittiDataGeneratorConfig.BATCH_SIZE,
            stage=stage,
            class_mapping=KittiDataGeneratorConfig.CLASS_MAPPING
        )


if __name__ == '__main__':
    print("Starting")
    KittiDataGeneratorConfig.BATCH_SIZE = 32
    db = KittiDataGeneratorConfig.build(Stage.VAL)
    print(len(db))
    print(db[0])

