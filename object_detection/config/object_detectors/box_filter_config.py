from backend.model.object_detection.box_filter import BoxFilter


class BoxFilterConfig:
    MIN_CLASS_PROB = 0.2
    NMS_IOU_THRESHOLD = 0.2

    @staticmethod
    def build(max_boxes_per_image, batch_size):
        return BoxFilter(
            min_class_prob=BoxFilterConfig.MIN_CLASS_PROB,
            nms_iou_threshold=BoxFilterConfig.NMS_IOU_THRESHOLD,
            max_boxes_per_image=max_boxes_per_image,
            batch_size=batch_size
        )
