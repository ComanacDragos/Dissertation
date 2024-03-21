from tqdm import tqdm

from backend.enums import Stage, DataType, LabelType
from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig


def count_boxes():
    all_counts = []
    KittiDataGeneratorConfig.BATCH_SIZE = 1  # 7481
    ds = KittiDataGeneratorConfig.build(Stage.ALL)
    for i in tqdm(range(len(ds)), total=len(ds)):
        boxes = ds[i][DataType.LABEL][0][LabelType.COORDINATES]
        all_counts.append(len(boxes))
    print(max(all_counts))
    # 21 for kitti


if __name__ == '__main__':
    count_boxes()
