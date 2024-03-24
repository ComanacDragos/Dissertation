from backend.model.object_detection.yolo_model import YOLOPostprocessing
from config.object_detectors.box_filter_config import BoxFilterConfig


class YOLOPostprocessingConfig:
    APPLY_ARGMAX = True
    BACKGROUND_PROB = 0.5

    @staticmethod
    def build(image_size, grid_size, anchors, max_boxes_per_image, batch_size):
        return YOLOPostprocessing(
            image_size=image_size,
            grid_size=grid_size,
            anchors=anchors,
            background_prob=YOLOPostprocessingConfig.BACKGROUND_PROB,
            box_filter=BoxFilterConfig.build(
                max_boxes_per_image=max_boxes_per_image,
                batch_size=batch_size
            )
        )
