from backend.model.object_detection.fcos_model import FCOSPostprocessing
from config.object_detectors.box_filter_config import BoxFilterConfig


class FCOSPostprocessingConfig:
    @staticmethod
    def build(image_size, max_boxes_per_image, batch_size):
        return FCOSPostprocessing(
            image_size=image_size,
            box_filter=BoxFilterConfig.build(
                max_boxes_per_image=max_boxes_per_image,
                batch_size=batch_size
            )
        )
