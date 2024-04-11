import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from backend.enums import LabelType, OutputType, ObjectDetectionOutputType
from backend.loss.yolo_loss import create_cell_grid
from backend.model.softmax import softmax


class FCOSPreprocessing:
    def __init__(self, image_size, no_classes, strides, thresholds):
        self.strides = strides  # e.g. (32, 16, 8)
        self.thresholds = thresholds  # e.g (128, 64)
        self.no_classes = no_classes
        self.image_size = np.asarray(image_size)  # h w

    def decode_gt(self, ground_truth):
        """
        Extracts from a single ground truth a tensor of scores, boxes and classes

        :param ground_truth: ground truth
        :return: scores, boxes, classes
        """
        pass

    def process_sample(self, classes, coordinates, stride, object_sizes_of_interest):
        grid_size = self.image_size // stride
        regression_targets = np.zeros((*grid_size, 4))
        classification_targets = np.zeros((*grid_size, self.no_classes))
        centerness_targets = np.zeros((*grid_size, 1))

        H, W = self.image_size

        locations = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1)
        remapped_locations = locations*stride + stride // 2

        xs = remapped_locations[..., 0]
        ys = remapped_locations[..., 1]

        l = xs[:, :, None] - coordinates[:, 0]
        t = ys[:, :, None] - coordinates[:, 1]
        r = coordinates[:, 2]- xs[:, :, None]
        b = coordinates[:, 3]- ys[:, :, None]

        regression_targets = np.stack([l, t, r, b], axis=2)

        is_in_boxes = np.min(regression_targets, axis=2) > 0

        return classification_targets, centerness_targets, regression_targets

    def __call__(self, labels):
        batch_classification_targets, batch_centerness_targets, batch_regression_targets = [], [], []
        for label in labels:
            classification_targets, centerness_targets, regression_targets = self.process_sample(label[LabelType.CLASS], label[LabelType.COORDINATES], 2, [0, 64])
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


class FCOSPostprocessing:
    def __init__(self, image_size, grid_size, anchors, background_prob, box_filter=None):
        self.image_size = image_size
        self.grid_size = grid_size
        self.background_prob = background_prob
        self.cell_grid = create_cell_grid(grid_size, len(anchors))
        cell_size = np.asarray(image_size) / np.asarray(grid_size)
        self.cell_size = cell_size[1], cell_size[0]

        self.anchors = np.asarray(anchors)
        self.box_filter = box_filter

    def _flatten(self, output):
        return tf.reshape(output, (-1, self.grid_size[0] * self.grid_size[1] * len(self.anchors), output.shape[-1]))

    def _decode_one_scale(self, scale_outputs):
        classification, centerness, regression = scale_outputs

        _, H, W, _ = classification.shape
        stride = self.image_size / H

        classification = K.sigmoid(classification)
        centerness = K.sigmoid(centerness)

        # K.mes

    def __call__(self, output):
        """
        Extracts from the output scores, boxes, and classes

        :param output: prediction made by model
        :return: scores, boxes, classes
        """
        xy = (K.sigmoid(output[..., :2]) + self.cell_grid) * self.cell_size
        wh = K.exp(output[..., 2:4]) * self.anchors
        boxes = tf.concat([xy - wh / 2, xy + wh / 2], axis=-1)
        # boxes = tf.concat([xy - 20, xy + 20], axis=-1)

        conf_scores = K.sigmoid(output[..., 4:5])
        classes = conf_scores * softmax(output[..., 5:])

        boxes = self._flatten(boxes)
        clip_values = self.image_size + self.image_size
        boxes = tf.clip_by_value(boxes, clip_value_min=0, clip_value_max=clip_values)

        classes = self._flatten(classes)

        best_classes_prob = tf.reduce_max(classes, axis=-1)
        best_classes = tf.argmax(classes, axis=-1)

        processed_outputs = {
            OutputType.COORDINATES: boxes.numpy(),
            OutputType.CLASS_PROBABILITIES: best_classes_prob.numpy(),
            OutputType.CLASS_LABEL: best_classes.numpy(),
            OutputType.ALL_CLASS_PROBABILITIES: classes.numpy()
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

        self.classification_1x1 = conv_generator(1, no_classes)
        self.centerness_1x1 = conv_generator(1, 1)
        self.regression_1x1 = conv_generator(1, 4)

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
            [4.1234, 5.4243, 7.8153, 8.7565]
        ]),
        LabelType.CLASS: np.asarray([
            [0, 1],
            [1, 0]
        ])
    }
    ]
    preprocessor = FCOSPreprocessing(
        (10, 20),
        2,
        (32, 16, 8),
        (128, 64)
    )

    processed_gt, true_bboxes = preprocessor(fake_gt * 8)
    print(processed_gt.shape)
    processed_gt = processed_gt[0]
    for i in range(processed_gt.shape[0]):
        for j in range(processed_gt.shape[1]):
            print(i, j)
            print(processed_gt[i, j, :, :])
            print()

    # _conf, _boxes, _classes = preprocessor.decode_gt(processed_gt)
    #
    # print(_boxes)
    # print(_classes)
