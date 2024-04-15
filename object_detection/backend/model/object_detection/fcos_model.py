import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from backend.enums import LabelType, OutputType, ObjectDetectionOutputType
from backend.loss.yolo_loss import create_cell_grid
from backend.model.softmax import softmax

INF = 2 ** 64


class FCOSPreprocessing:
    def __init__(self, image_size, no_classes, strides, thresholds):
        """

        :param image_size: h, w
        :param no_classes: int
        :param strides:  e.g. (8, 16, 32)
        :param thresholds: e.g (64, 128)
        """
        self.strides = strides
        self.thresholds = [0, *thresholds, INF]
        self.no_classes = no_classes
        self.image_size = np.asarray(image_size)

    def decode_gt(self, ground_truth):
        """
        Extracts from a single ground truth a tensor of scores, boxes and classes

        :param ground_truth: ground truth
        :return: scores, boxes, classes
        """
        pass

    def _process_sample(self, classes, coordinates, stride, size_of_interest):
        grid_size = self.image_size // stride
        H, W = grid_size

        if len(coordinates) == 0:
            return np.zeros((H, W, self.no_classes)), np.zeros((H, W)), np.zeros((H, W, 4))

        h_cells, w_cells = np.arange(H), np.arange(W)
        cells = np.stack(np.meshgrid(h_cells, w_cells, indexing='ij'), axis=-1)
        remapped_cells = cells * stride + stride // 2

        xs = remapped_cells[..., 1]
        ys = remapped_cells[..., 0]

        l = xs[:, :, None] - coordinates[:, 0]
        t = ys[:, :, None] - coordinates[:, 1]
        r = coordinates[:, 2] - xs[:, :, None]
        b = coordinates[:, 3] - ys[:, :, None]

        regression_targets_to_gt = np.stack([l, t, r, b], axis=2)

        is_in_boxes = np.min(regression_targets_to_gt, axis=2) > 0

        max_reg_targets = np.max(regression_targets_to_gt, axis=2)
        coordinates_belong_to_scale = (max_reg_targets >= size_of_interest[0]) & (max_reg_targets < size_of_interest[1])

        # choose bboxes with minimum area
        areas = (coordinates[:, 2] - coordinates[:, 0]) * (coordinates[:, 3] - coordinates[:, 1])
        cells_to_gt_areas = areas[None].repeat(W, 0)[None].repeat(H, 0)
        cells_to_gt_areas[~is_in_boxes] = INF
        cells_to_gt_areas[~coordinates_belong_to_scale] = INF

        # map cells to gt
        cells_to_min_area = np.min(cells_to_gt_areas, axis=-1)
        cells_to_gt_inds = np.argmin(cells_to_gt_areas, axis=-1)

        # compute final targets
        regression_targets = regression_targets_to_gt[h_cells[:, None], w_cells[None, :], :, cells_to_gt_inds]
        regression_targets[cells_to_min_area == INF] = 0

        classification_targets = np.zeros((*grid_size, self.no_classes))
        classification_targets[h_cells[:, None], w_cells[None, :], np.argmax(classes, axis=-1)[cells_to_gt_inds]] = 1
        classification_targets[cells_to_min_area == INF] = 0

        eps = 1e-9
        left_right = regression_targets[:, :, [0, 2]] + eps
        top_bottom = regression_targets[:, :, [1, 3]] + eps
        centerness_targets = (np.min(left_right, axis=-1) / np.max(left_right, axis=-1)) * (
                np.min(top_bottom, axis=-1) / np.max(top_bottom, axis=-1))
        centerness_targets = np.sqrt(centerness_targets)
        centerness_targets[cells_to_min_area == INF] = 0

        return classification_targets, centerness_targets, regression_targets

    def _process_scale(self, labels, stride, size_of_interest):
        batch_classification_targets, batch_centerness_targets, batch_regression_targets = [], [], []
        for label in labels:
            classification_targets, centerness_targets, regression_targets = self._process_sample(
                label[LabelType.CLASS],
                label[LabelType.COORDINATES],
                stride,
                size_of_interest
            )
            batch_classification_targets.append(classification_targets)
            batch_centerness_targets.append(centerness_targets)
            batch_regression_targets.append(regression_targets)

        batch_classification_targets = np.stack(batch_classification_targets)
        batch_centerness_targets = np.stack(batch_centerness_targets)
        batch_regression_targets = np.stack(batch_regression_targets)

        return (
            tf.convert_to_tensor(batch_classification_targets, dtype=tf.float32),
            tf.convert_to_tensor(batch_centerness_targets, dtype=tf.float32),
            tf.convert_to_tensor(batch_regression_targets, dtype=tf.float32),
        )

    def __call__(self, labels):
        return {
            stride: self._process_scale(labels, stride, self.thresholds[i:i + 2])
            for i, stride in enumerate(self.strides)
        }


class FCOSPostprocessing:
    def __init__(self, image_size, box_filter=None):
        self.image_size = image_size
        self.box_filter = box_filter

    def _flatten(self, output):
        return tf.reshape(output, (-1, self.grid_size[0] * self.grid_size[1] * len(self.anchors), output.shape[-1]))

    def _decode_one_scale(self, stride, classification, centerness, regression):
        N, H, W, no_classes = classification.shape

        # prepare class probabilities
        probabilities = K.sigmoid(classification)
        centerness = K.sigmoid(centerness)
        probabilities = probabilities * centerness

        # compute grid
        h_cells, w_cells = tf.range(H), tf.range(W)
        cells = tf.stack(tf.meshgrid(h_cells, w_cells, indexing='ij'), axis=-1)
        remapped_cells = cells * stride + stride // 2
        remapped_cells = tf.repeat(remapped_cells[None, ...], N, axis=0)
        remapped_cells = tf.cast(remapped_cells, regression.dtype)

        regression = tf.math.exp(regression)
        boxes = tf.stack([
            remapped_cells[..., 0] - regression[..., 0],
            remapped_cells[..., 1] - regression[..., 1],
            remapped_cells[..., 0] + regression[..., 2],
            remapped_cells[..., 1] + regression[..., 3],
        ], axis=-1)
        img_H, img_W = self.image_size
        boxes = tf.clip_by_value(boxes, clip_value_min=0, clip_value_max=[img_W, img_H, img_W, img_H])

        # flatten
        probabilities = tf.reshape(probabilities, (N, -1, no_classes))
        boxes = tf.reshape(boxes, (N, -1, 4))

        return probabilities, boxes

    def __call__(self, output):
        """
        Extracts from the output scores, boxes, and classes

        :param output: prediction made by model
        :return: scores, boxes, classes
        """
        all_probs = []
        all_boxes = []
        for stride, scale_outputs in output.items():
            probs, boxes = self._decode_one_scale(stride, *scale_outputs)
            all_probs.append(probs)
            all_boxes.append(boxes)
        all_probs = tf.concat(all_probs, axis=1)
        all_boxes = tf.concat(all_boxes, axis=1)

        best_classes_prob = tf.reduce_max(all_probs, axis=-1)
        best_classes = tf.argmax(all_probs, axis=-1)

        processed_outputs = {
            OutputType.COORDINATES: all_boxes.numpy(),
            OutputType.CLASS_PROBABILITIES: best_classes_prob.numpy(),
            OutputType.CLASS_LABEL: best_classes.numpy(),
            OutputType.ALL_CLASS_PROBABILITIES: all_probs.numpy()
        }

        outputs = {
            ObjectDetectionOutputType.BEFORE_FILTERING: processed_outputs
        }
        if self.box_filter:
            outputs[ObjectDetectionOutputType.AFTER_FILTERING] = self.box_filter(processed_outputs)
        return outputs


class FCOSHead:
    def __init__(self, no_classes, conv_generator, classification_branch, regression_branch):
        self.no_classes = no_classes
        self.classification_branch = classification_branch
        self.regression_branch = regression_branch

        self.classification_1x1 = conv_generator(1, no_classes)()
        self.centerness_1x1 = conv_generator(1, 1)()
        self.regression_1x1 = conv_generator(1, 4)()

    def __call__(self, inputs):
        classification_outputs = self.classification_branch(inputs)
        regression_outputs = self.regression_branch(inputs)

        return (
            self.classification_1x1(classification_outputs),
            self.centerness_1x1(classification_outputs),
            self.regression_1x1(regression_outputs),
        )


if __name__ == '__main__':
    fake_gt = [{
        LabelType.COORDINATES: np.asarray([
            [1.7, 2.3, 3.23, 4.43],
            [4.1234, 5.4243, 7.8153, 8.7565],
            [10, 5, 19, 9],
            [12, 5, 17, 9],
            [12, 5, 18, 9]
        ]),
        LabelType.CLASS: np.asarray([
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
        ])
    }
    ]
    preprocessor = FCOSPreprocessing(
        (10*32, 20*32),
        2,
        (8, 16, 32),
        (64, 128)
    )

    scales = preprocessor(fake_gt * 8)
    print(len(scales))
    for _stride, scale in scales.items():
        print(_stride, [x.shape for x in scale])

    post_process = FCOSPostprocessing((10, 20), min_class_prob=0.5)
    post_process({
        2: (
            tf.convert_to_tensor(np.random.random((8, 5, 10, 3))),
            tf.convert_to_tensor(np.random.random((8, 5, 10, 1))),
            tf.convert_to_tensor(np.random.random((8, 5, 10, 4)))
        )
    })