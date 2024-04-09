import numpy as np
from tqdm import tqdm

from backend.enums import Stage, DataType, LabelType
from backend.utils import to_json, open_json
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig
from sklearn.cluster import KMeans


def generate_boxes():
    output = "outputs/kitti_boxes_v2.json"
    KittiDataGeneratorConfig.BATCH_SIZE = 1  # 7481
    ds = KittiDataGeneratorConfig.build(Stage.ALL)
    all_boxes = []
    for i in tqdm(range(len(ds)), total=len(ds)):
        boxes = ds[i][DataType.LABEL][0][LabelType.COORDINATES]
        if len(boxes) == 0:
            continue
        all_boxes.append(boxes)

    to_json(np.concatenate(all_boxes).tolist(), output)


def generate_anchors():
    output = "scripts/kitti_anchors_v2.json"
    boxes = np.asarray(open_json("outputs/kitti_boxes_v2.json"))
    top_left = boxes[:, :2]
    bottom_right = boxes[:, 2:]
    wh = bottom_right - top_left
    all_anchors = {}
    for n_clusters in range(2, 11):
        print(n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(wh)
        all_anchors[n_clusters] = kmeans.cluster_centers_.tolist()

    print(all_anchors)
    to_json(all_anchors, output)


if __name__ == '__main__':
    generate_boxes()
    generate_anchors()