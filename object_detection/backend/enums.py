from enum import Enum, auto


class Stage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    ALL = 'all'


class DataType(Enum):
    IDENTIFIER = auto()
    IMAGE = auto()
    LABEL = auto()
    PREDICTION = auto()


class LabelType(Enum):
    COORDINATES = auto()
    CLASS = auto()


class OutputType(Enum):
    COORDINATES = auto()
    CLASS_PROBABILITIES = auto()
    CLASS_LABEL = auto()
    ALL_CLASS_PROBABILITIES = auto()


class ObjectDetectionOutputType(Enum):
    BEFORE_FILTERING = auto()
    AFTER_FILTERING = auto()
