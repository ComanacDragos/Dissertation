import numpy as np
from overrides import overrides

from backend.enums import LabelType
from backend.data_generator.object_detection_data_generator import ObjectDetectionDataGenerator


class PublicTransportDataGenerator(ObjectDetectionDataGenerator):
    @overrides
    def load_label(self, relative_label_path):
        """
        Loads the OID label, each file contains several rows in the following format:
        <object_type> <left> <top> <right> <bottom>
        """
        classes = []
        coordinates = []
        with open(self.root / relative_label_path) as f:
            for line in f.readlines():
                tokens = line.split()
                cls = " ".join(tokens[:-4])
                if cls not in self.class_mapping:
                    continue
                left, top, right, bottom = tokens[-4:]
                classes.append(self.one_hot_encode_class(cls))
                coordinates.append([float(left), float(top), float(right), float(bottom)])

        return {
            LabelType.COORDINATES: np.asarray(coordinates),
            LabelType.CLASS: np.asarray(classes)
        }
