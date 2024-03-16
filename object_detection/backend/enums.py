from enum import Enum, auto


class Stage(Enum):
    TRAIN = 'train'
    VAL = 'val'


class DataType(Enum):
    IDENTIFIER = auto()
    IMAGE = auto()
    LABEL = auto()
    PREDICTION = auto()


class LabelType(Enum):
    COORDINATES = auto()
    CLASS = auto()
