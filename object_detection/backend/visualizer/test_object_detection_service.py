import unittest

import numpy as np
import pandas as pd

from backend.enums import DataType, LabelType, ObjectDetectionOutputType, OutputType
from backend.visualizer.object_detection_service import ObjectDetectionService


class MockDataGenerator:
    def __init__(self, include_labels):
        self.include_labels = include_labels
        self.data = pd.DataFrame.from_dict({"image": []})

    def __getitem__(self, index):
        data = {
            DataType.IMAGE: np.zeros((1, 160, 512, 3))
        }
        if self.include_labels:
            data[DataType.LABEL] = [{
                LabelType.CLASS: [],
                LabelType.COORDINATES: []
            }]
        return data


class MockModel:
    def __call__(self, image):
        return {
            ObjectDetectionOutputType.AFTER_FILTERING: {
                OutputType.COORDINATES: [[]],
                OutputType.CLASS_LABEL: [[]],
                OutputType.CLASS_PROBABILITIES: [[]]
            }
        }


class TestStringMethod(unittest.TestCase):
    def test_gt_only(self):
        service = ObjectDetectionService(MockDataGenerator(True))
        img = service[0]
        assert img.shape == (160, 512, 3)

    def test_pred_only(self):
        service = ObjectDetectionService(MockDataGenerator(False), MockModel())
        img = service[0]
        assert img.shape == (160, 512, 3)

    def test_pred_and_gt(self):
        service = ObjectDetectionService(MockDataGenerator(True), MockModel())
        img = service[0]
        assert img.shape == (340, 512, 3)
