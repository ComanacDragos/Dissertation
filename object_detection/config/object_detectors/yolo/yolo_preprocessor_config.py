from backend.model.yolo_model import YOLOPreprocessing


class YOLOPreprocessingConfig:
    MAX_BOXES_PER_IMAGE = 21

    @staticmethod
    def build(image_size, grid_size, anchors, no_classes):
        return YOLOPreprocessing(
            image_size,
            grid_size,
            anchors,
            no_classes,
            YOLOPreprocessingConfig.MAX_BOXES_PER_IMAGE
        )
