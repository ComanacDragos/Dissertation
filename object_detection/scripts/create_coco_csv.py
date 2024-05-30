import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import json
from collections import Counter
from pprint import pprint

root = Path(r"C:\Users\Dragos\datasets\coco")

def create_csv():
    final_data = {
        "image": [],
        "label": [],
        "stage": [],
    }

    for stage in ["train", "val"]:
        path = fr"C:\Users\Dragos\datasets\coco\annotations\instances_{stage}2017.json"
        print(f"Loading {path}...")
        data = json.load(open(path))
        for image_data in tqdm(data['images'], total=len(data['images'])):
            image_name = image_data['file_name']
            label_name = image_data['id']

            image_path = Path(f"{stage}2017") / image_name

            final_data["image"].append(str(image_path))
            final_data["label"].append(str(label_name))
            final_data["stage"].append(stage)

    pd.DataFrame.from_dict(final_data).to_csv(f"csvs/coco.csv", index=False)

def create_anno():
    final_data = {

    }
    labels = json.load(open(root / "categories.json"))

    for stage in ["val", 'train']:
        path = fr"C:\Users\Dragos\datasets\coco\annotations\instances_{stage}2017.json"
        print(f"Loading {path}...")
        data = json.load(open(path))
        print(data.keys())
        for anno_data in tqdm(data['annotations'], total=len(data['annotations'])):
            image_id = anno_data['image_id']
            bbox = anno_data['bbox']
            # print(anno_data['category_id'])
            category_id = anno_data['category_id']
            anno_id = anno_data['id']
            box_data = {
                    'bbox': bbox,
                    'category_id': category_id,
                    'anno_id': anno_id
                }

            if image_id in final_data:
                final_data[image_id].append(box_data)
            else:
                final_data[image_id] = [box_data]
    print(len(final_data))
    json.dump(final_data, open(str(root / "anno.json"), 'w'))

def check_shapes():
    shapes = []
    for stage in ["val", 'train']:
        path = fr"C:\Users\Dragos\datasets\coco\annotations\instances_{stage}2017.json"

        print(f"Loading {path}...")
        data = json.load(open(path))

        images = data['images']
        for img in tqdm(images, total=len(images)):
            shapes.append((img['height'], img['width']))

    pprint(Counter(shapes))

    shapes = np.asarray(shapes)
    h, w = shapes[:, 0], shapes[:, 1]
    print("Height:", np.min(h), np.max(h))
    print("Width:", np.min(w), np.max(w))

if __name__ == '__main__':
    create_csv()
    # create_anno()
    # check_shapes()