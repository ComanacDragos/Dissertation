from backend.model.object_detection.fcos_model import FCOSPreprocessing


class FCOSPreprocessingConfig:

    @staticmethod
    def build(image_size, no_classes, strides, thresholds):
        return FCOSPreprocessing(
            image_size,
            no_classes,
            strides,
            thresholds,
        )
