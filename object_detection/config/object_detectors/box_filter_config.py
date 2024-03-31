from backend.model.object_detection.box_filter import BoxFilter


class BoxFilterConfig:
    MIN_OBJ_PROB = 0.2
    MIN_CLASS_PROB = 0.1
    NMS_IOU_THRESHOLD = 0.5

    @staticmethod
    def build(max_boxes_per_image, batch_size):
        return BoxFilter(
            min_obj_prob=BoxFilterConfig.MIN_OBJ_PROB,
            min_class_prob=BoxFilterConfig.MIN_CLASS_PROB,
            nms_iou_threshold=BoxFilterConfig.NMS_IOU_THRESHOLD,
            max_boxes_per_image=max_boxes_per_image,
            batch_size=batch_size
        )
