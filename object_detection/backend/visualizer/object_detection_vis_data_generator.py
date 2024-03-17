import numpy as np
from PIL import Image

from backend.data_generator.object_detection_data_generator import ObjectDetectionDataGenerator
from backend.enums import DataType, LabelType


class aug_category:

    def __init__(self, categories):
        self.category = categories
        self.combo_list = categories.copy()
        self.combo_list.insert(0, 'All')
        self.all = True


class ObjectDetectionVisDataGenerator:
    def __init__(self, source_data_generator: ObjectDetectionDataGenerator):
        self.source_data_generator = source_data_generator
        self.aug_category = aug_category(source_data_generator.labels)
        self.img_list = list(source_data_generator.data.loc[:, "image"])
        self.det_file = ''
        self.has_anno = True
        self.mask = False

    def get_img_by_index(self, index):
        return Image.fromarray(self.source_data_generator[index][DataType.IMAGE][0])

    def get_singleImg_gt(self, name):
        """
        Return GT for a specific image in the following format: cls, xmin, ymin, width, height
        """
        data = self.source_data_generator[self.img_list.index(name)][DataType.LABEL][0]

        classes = [self.source_data_generator.labels[x] for x in np.argmax(data[LabelType.CLASS], axis=-1)]
        coordinates = [[int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)] for xmin, ymin, xmax, ymax in
                       data[LabelType.COORDINATES].tolist()]

        return [[cls, *coords] for cls, coords in zip(classes, coordinates)]

    def get_img_by_name(self, name):
        return self.get_img_by_index(self.img_list.index(name))
