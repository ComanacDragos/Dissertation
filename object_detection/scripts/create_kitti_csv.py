import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

root = Path(r"C:\Users\Dragos\datasets\KITTI")

data = {
    "image": [],
    "label": [],
    "stage": [],
}

train_prob = 0.8

image_paths = list(os.listdir(root / "images"))
for image_name in tqdm(image_paths, total=len(image_paths)):
    label_name = image_name.split(".")[0] + ".txt"

    label_path = Path("labels") / label_name
    if not os.path.exists(root / label_path):
        print(f"Bad path: {image_name} { label_path}")
        continue

    image_path = Path("images") / image_name

    data["image"].append(str(image_path))
    data["label"].append(str(label_path))
    data["stage"].append("train" if np.random.random() < train_prob else "val")

pd.DataFrame.from_dict(data).to_csv("csvs/kitti.csv", index=False)

