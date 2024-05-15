from pathlib import Path

import cv2
import pandas as pd

from backend.enums import DataType


class InferenceDataGenerator:
    def __init__(self, root, csv_path, labels, image_preprocessing=None):
        self.root = Path(root)
        self.labels = labels
        self.data = pd.read_csv(csv_path)
        self.image_preprocessing = image_preprocessing

    def load_image(self, relative_image_path):
        img = cv2.imread(str(self.root / relative_image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_preprocessing:
            img = self.image_preprocessing(image=img, bboxes=[], class_labels=[])['image']
        return img

    def __getitem__(self, index):
        img_path = self.data.iloc[index]['image']
        return {DataType.IMAGE: [self.load_image(img_path)]}
