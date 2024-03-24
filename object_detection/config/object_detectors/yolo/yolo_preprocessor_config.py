from backend.model.object_detection.yolo_model import YOLOPreprocessing


class YOLOPreprocessingConfig:

    @staticmethod
    def build(image_size, grid_size, anchors, no_classes, max_boxes_per_image):
        return YOLOPreprocessing(
            image_size,
            grid_size,
            anchors,
            no_classes,
            max_boxes_per_image
        )
