import numpy as np
from overrides import overrides

from backend.enums import LabelType
from backend.data_generator.object_detection_data_generator import ObjectDetectionDataGenerator


class KittiDataGenerator(ObjectDetectionDataGenerator):
    @overrides
    def load_label(self, relative_label_path):
        """
        Loads the kitti label, each file contains several rows in the following format:
        <object_type> <truncation> <occlusion> <alpha> <left> <top> <right> <bottom> <height> <width> <length> <x> <y> <z> <rotation_y>
        """
        classes = []
        coordinates = []
        with open(self.root / relative_label_path) as f:
            for line in f.readlines():
                cls, _, _, _, left, top, right, bottom, _, _, _, _, _, _, _ = line.split()
                if cls not in self.class_mapping:
                    continue
                classes.append(self.one_hot_encode_class(cls))
                coordinates.append([left, top, right, bottom])

        return {
            LabelType.COORDINATES: np.asarray(coordinates),
            LabelType.CLASS: np.asarray(classes)
        }
