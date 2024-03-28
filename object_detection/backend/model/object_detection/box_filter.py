import tensorflow as tf
import numpy as np
from backend.enums import OutputType


class BoxFilter:
    def __init__(self, min_obj_prob, min_class_prob, nms_iou_threshold, max_boxes_per_image, batch_size):
        self.min_obj_prob = min_obj_prob
        self.min_class_prob = min_class_prob
        self.nms_iou_threshold = nms_iou_threshold
        self.max_boxes_per_image = max_boxes_per_image
        self.batch_size = batch_size

    def __call__(self, outputs):
        class_probs = outputs[OutputType.ALL_CLASS_PROBABILITIES]
        boxes = outputs[OutputType.COORDINATES]

        nms_boxes = tf.expand_dims(boxes, axis=2)
        nms_boxes, nms_scores, nms_classes, nms_valid = tf.image.combined_non_max_suppression(
            nms_boxes, class_probs, self.max_boxes_per_image, self.max_boxes_per_image * self.batch_size,
            iou_threshold=self.nms_iou_threshold,
            score_threshold=self.min_class_prob, clip_boxes=False)

        def filter_tensor(tensor, num_valids):
            return [x[:num_valid] for x, num_valid in zip(tensor.numpy(), num_valids.numpy())]

        return {
            OutputType.COORDINATES: filter_tensor(nms_boxes, nms_valid),
            OutputType.CLASS_PROBABILITIES: filter_tensor(nms_scores, nms_valid),
            OutputType.CLASS_LABEL: filter_tensor(nms_classes, nms_valid),
        }
