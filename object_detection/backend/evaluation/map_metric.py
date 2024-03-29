import numpy as np

from backend.enums import LabelType, OutputType
from .from_mmdet.mean_ap import eval_map
from collections import OrderedDict


class MeanAP:
    def __init__(self, labels, iou_threshold, scale_ranges=None):
        self.labels = labels
        self.iou_threshold = iou_threshold
        self.scale_ranges = scale_ranges

    def __call__(self, logs):
        # prepare gts
        annotations = [
            {
                'bboxes': gt[LabelType.COORDINATES] if len(gt[LabelType.COORDINATES]) > 0 else np.zeros((0, 4)),
                'labels': np.argmax(gt[LabelType.CLASS], axis=-1) if len(gt[LabelType.CLASS]) > 0 else np.zeros((0,))
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

        mAP, aps = eval_map(
            det_results,
            annotations,
            scale_ranges=self.scale_ranges,
            iou_thr=self.iou_threshold,
            labels=self.labels
        )

        metrics = OrderedDict()
        metrics['mAP'] = mAP

        for i, label in enumerate(self.labels):
            metrics[f'{label}_AP'] = aps[i]['ap']

        return metrics
