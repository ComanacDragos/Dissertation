import cv2
import pandas as pd
from PIL import Image

from backend.visualizer.aug_category import aug_category
from pathlib import Path


class InferenceDataGenerator:
    def __init__(self, root, csv_path, labels, augmentations=None):
        self.root = Path(root)
        self.labels = labels
        self.aug_category = aug_category(labels)
        self.img_list = list(pd.read_csv(csv_path).loc[:, "image"])
        self.det_file = ''
        self.has_anno = False
        self.mask = False
        self.augmentations = augmentations

    def load_image(self, relative_image_path):
        img = cv2.imread(str(self.root / relative_image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augmentations:
            img = self.augmentations(image=img,  bboxes=[], class_labels=[])['image']
        return img

    def get_img_by_index(self, index):
        return Image.fromarray(self.load_image(self.img_list[index]))

    def get_singleImg_gt(self, name):
        """
        Returns an empty list since there are no labels during inference
        """
        return []

    def get_img_by_name(self, name):
        return self.get_img_by_index(self.img_list.index(name))
