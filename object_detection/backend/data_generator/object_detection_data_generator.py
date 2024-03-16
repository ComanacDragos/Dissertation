from abc import abstractmethod

import cv2
from overrides import overrides

from backend.enums import DataType
from backend.data_generator.generic_data_generator import GenericDataGenerator
import numpy as np


class ObjectDetectionDataGenerator(GenericDataGenerator):
    def __init__(self, class_mapping, **kwargs):
        super().__init__(**kwargs)
        self.class_mapping = class_mapping
        self.labels = sorted(class_mapping.values())

    def load_image(self, relative_image_path):
        img = cv2.imread(str(self.root / relative_image_path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def one_hot_encode_class(self, cls: str):
        remapped_class = self.class_mapping[cls]
        encoding = np.zeros(len(self.labels))
        encoding[self.labels.index(remapped_class)] = 1

        return encoding
        
    @abstractmethod
    def load_label(self, relative_label_path):
        pass

    @overrides()
    def create_batch(self, batch_data):
        image_paths = [sample['image'] for sample in batch_data]
        label_paths = [sample['label'] for sample in batch_data]

        images = [self.load_image(image_path) for image_path in image_paths]
        labels = [self.load_label(label_path) for label_path in label_paths]

        return {
            DataType.IDENTIFIER: image_paths,
            DataType.IMAGE: images,
            DataType.LABEL: labels,
        }
