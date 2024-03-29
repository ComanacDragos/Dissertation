from backend.enums import DataType, LabelType, OutputType
from .from_mmdet.mean_ap import eval_map
import numpy as np

class MeanAP:
    def __init__(self, labels, iou_threshold, scale_ranges=None):
        self.labels = labels
        self.iou_threshold = iou_threshold
        self.scale_ranges = scale_ranges

    def __call__(self, logs):
        # prepare gts
        annotations = [
            {
                'bboxes': gt[LabelType.COORDINATES],
                'labels': np.argmax(gt[LabelType.CLASS], axis=-1) if len(gt[LabelType.CLASS]) > 0 else np.zeros((0, 4))
            }
            for gt in logs['labels']
        ]

        # prepare dets:
        det_results = []
        for prediction in logs['predictions']:
            det = []
            classes = prediction[OutputType.CLASS_LABEL]
            boxes = prediction[OutputType.COORDINATES]
            for cls_id in range(len(self.labels)):
                det.append(boxes[classes == cls_id])
            det_results.append(det)

        return eval_map(
            det_results,
            annotations,
            scale_ranges=self.scale_ranges,
            iou_thr=self.iou_threshold,
            labels=self.labels
        )



