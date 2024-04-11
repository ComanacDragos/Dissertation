from enum import Enum, auto


class Stage(str, Enum):
    TRAIN = 'train'
    VAL = 'val'
    ALL = 'all'


class DataType(str, Enum):
    IDENTIFIER = 'IDENTIFIER'
    IMAGE = 'IMAGE'
    LABEL = 'LABEL'
    PREDICTION = 'PREDICTION'


class LabelType(str, Enum):
    COORDINATES = 'COORDINATES'
    CLASS = 'CLASS'


class OutputType(str, Enum):
    COORDINATES = 'COORDINATES'
    CLASS_PROBABILITIES = 'CLASS_PROBABILITIES'
    CLASS_LABEL = 'CLASS_LABEL'
    ALL_CLASS_PROBABILITIES = 'ALL_CLASS_PROBABILITIES'


class ObjectDetectionOutputType(str, Enum):
    BEFORE_FILTERING = 'BEFORE_FILTERING'
    AFTER_FILTERING = 'AFTER_FILTERING'
