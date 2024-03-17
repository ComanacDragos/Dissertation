import os
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from backend.enums import Stage, DataType
from backend.utils import to_json, open_json
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig


def labels_stats():
    root = Path(r"C:\Users\Dragos\datasets\KITTI\labels")

    classes = []
    paths = list(os.listdir(root))
    for label_name in tqdm(paths, total=len(paths)):
        with open(root / str(label_name)) as f:
            for line in f.readlines():
                tokens = line.split()

                classes.append(tokens[0])

    classes_freq = Counter(classes)
    print("CLASSES:", classes_freq.keys())
    print("Frequency:", classes_freq)


def generate_images_stats():
    output = "scripts/image_stats.json"
    KittiDataGeneratorConfig.BATCH_SIZE = 1
    ds = KittiDataGeneratorConfig.build(Stage.ALL)
    stats = {}
    for i in tqdm(range(len(ds)), total=len(ds)):
        data = ds[i]
        identifier = data[DataType.IDENTIFIER][0]
        image = data[DataType.IMAGE][0]
        stats[identifier] = {
            "shape": np.shape(image),
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "min": float(np.min(image)),
            "max": float(np.max(image))
        }

    to_json(stats, output)


def process_stats():
    data = open_json("scripts/image_stats.json")

    shapes = [tuple(x['shape']) for _, x in data.items()]

    print(Counter(shapes))


if __name__ == '__main__':
    #generate_images_stats()
    process_stats()