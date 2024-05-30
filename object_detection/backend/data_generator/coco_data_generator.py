import json

import numpy as np
from overrides import overrides

from backend.data_generator.object_detection_data_generator import ObjectDetectionDataGenerator
from backend.enums import LabelType


class COCODataGenerator(ObjectDetectionDataGenerator):
    def __init__(self, category_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno = json.load(open(self.root / "anno.json"))
        self.categories = json.load(open(self.root / "categories.json"))
        self.category_type = category_type

    @overrides
    def load_label(self, relative_label_path):
        """
        Loads the COCO label
        """
        classes = []
        coordinates = []
        if str(relative_label_path) in self.anno:
            for box in self.anno[str(relative_label_path)]:
                label = self.categories[str(box['category_id'])][self.category_type]
                classes.append(self.one_hot_encode_class(label))
                left, top, w, h = box['bbox']
                right = left + w
                bottom = top + h
                coordinates.append([left, top, right, bottom])

        return {
            LabelType.COORDINATES: np.asarray(coordinates),
            LabelType.CLASS: np.asarray(classes)
        }


if __name__ == '__main__':
    path = r"C:\Users\Dragos\datasets\coco\annotations\instances_val2017.json"
    data = json.load(open(path))
    a = 0
