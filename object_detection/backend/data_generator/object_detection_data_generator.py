from abc import abstractmethod

import cv2
import numpy as np
from overrides import overrides

from backend.data_generator.generic_data_generator import GenericDataGenerator
from backend.enums import DataType, LabelType


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

    def load_sample(self, sample):
        image = self.load_image(sample['image'])
        label = self.load_label(sample['label'])

        if self.augmentations:
            augmented = self.augmentations(image=image, bboxes=label[LabelType.COORDINATES],
                                           class_labels=label[LabelType.CLASS])
            image = augmented['image']
            label = {
                LabelType.CLASS: np.asarray(augmented['class_labels']),
                LabelType.COORDINATES: np.asarray(augmented['bboxes'])
            }
        return {
            DataType.IDENTIFIER: sample['image'],
            DataType.IMAGE: image,
            DataType.LABEL: label,
        }

    @overrides()
    def create_batch(self, batch_data):
        samples = [self.load_sample(sample) for sample in batch_data]

        return {
            DataType.IDENTIFIER: [sample[DataType.IDENTIFIER] for sample in samples],
            DataType.IMAGE: np.asarray([sample[DataType.IMAGE] for sample in samples]),
            DataType.LABEL: [sample[DataType.LABEL] for sample in samples],
        }
