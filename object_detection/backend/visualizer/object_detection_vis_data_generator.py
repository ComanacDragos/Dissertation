from backend.data_generator.object_detection_data_generator import ObjectDetectionDataGenerator
from backend.enums import DataType


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
        self.img_list = source_data_generator.data.loc[:, "image"]


    def get_img_by_index(self, index):
        return self.source_data_generator[index][DataType.IMAGE][0]

