import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from collections import Counter

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
