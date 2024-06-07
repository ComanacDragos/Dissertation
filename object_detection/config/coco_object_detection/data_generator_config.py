import json
from pathlib import Path

import albumentations as A

from backend.data_generator import COCODataGenerator
from backend.enums import Stage, DataType


def generate_categories(root, cat_type):
    return {x[cat_type]: x[cat_type] for x in json.load(open(Path(root) / "categories.json")).values()}


class COCODataGeneratorConfig:
    ROOT = r"C:\Users\Dragos\datasets\coco"
    CSV_PATH = "csvs/coco.csv"
    BATCH_SIZE = 8
    CATEGORY_TYPE = "name"

    CLASS_MAPPING = generate_categories(ROOT, CATEGORY_TYPE)
    MAX_BOXES_PER_IMAGE = 91 * 4

    LABELS = sorted(set(CLASS_MAPPING.values()))
    # IMAGE_SHAPE = (480, 640, 3)
    # INPUT_SHAPE = (480, 640, 3)
    INPUT_SHAPE = (320, 416, 3)
    SHUFFLE = True

    BBOX_PARAMS = A.BboxParams(format='pascal_voc', min_area=300, min_visibility=0.1,
                               label_fields=['class_labels'])
    IMAGE_PREPROCESSOR = A.Compose([
        A.Resize(
            width=INPUT_SHAPE[1],
            height=INPUT_SHAPE[0]
        )
    ], bbox_params=BBOX_PARAMS)

    @staticmethod
    def build(stage: Stage):
        augmentations = A.Compose([
            A.HorizontalFlip(),
            A.Cutout()
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=300, min_visibility=0.1,
                                    label_fields=['class_labels']))
        return COCODataGenerator(
            root=COCODataGeneratorConfig.ROOT,
            csv_path=COCODataGeneratorConfig.CSV_PATH,
            batch_size=COCODataGeneratorConfig.BATCH_SIZE,
            stage=stage,
            class_mapping=COCODataGeneratorConfig.CLASS_MAPPING,
            labels=COCODataGeneratorConfig.LABELS,
            augmentations=augmentations,
            image_preprocessor=COCODataGeneratorConfig.IMAGE_PREPROCESSOR,
            shuffle=COCODataGeneratorConfig.SHUFFLE,
            category_type=COCODataGeneratorConfig.CATEGORY_TYPE
        )


if __name__ == '__main__':
    print("Starting")
    # COCODataGeneratorConfig.BATCH_SIZE = 32
    # db = COCODataGeneratorConfig.build(Stage.VAL)
    # print(len(db))
    # data = db[0]
    # print(f"Images shape {data[DataType.IMAGE].shape}")

    from backend.utils import set_seed
    set_seed(0)
    COCODataGeneratorConfig.BATCH_SIZE = 1
    COCODataGeneratorConfig.SHUFFLE = False

    db = COCODataGeneratorConfig.build(Stage.ALL)

    from tqdm import tqdm
    for i in tqdm(range(len(db)), total=len(db)):
        try:
            data = db[i]
        except ValueError:
            continue