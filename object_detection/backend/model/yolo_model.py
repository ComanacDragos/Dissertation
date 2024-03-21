import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Reshape

from backend.enums import LabelType
from backend.loss.yolo_loss import create_cell_grid
from backend.model.softmax import softmax
from backend.utils import iou


class YOLOPreprocessing:
    def __init__(self, image_size, grid_size, anchors, no_classes, max_boxes_per_image):
        self.anchors = np.asarray(anchors) # wh format
        self.no_classes = no_classes
        self.image_size = np.asarray(image_size)
        self.grid_size = np.asarray(grid_size)
        self.max_boxes_per_image = max_boxes_per_image

    def decode_gt(self, ground_truth):
        """
        Extracts from a single ground truth a tensor of scores, boxes and classes

        :param ground_truth: ground truth
        :return: scores, boxes, classes
        """
        conf_scores, boxes, classes = [], [], []
        cell_size = self.image_size / self.grid_size
        for cx in range(self.grid_size[1]):
            for cy in range(self.grid_size[0]):
                for a in range(len(self.anchors)):
                    anchor = ground_truth[cy][cx][a]
                    if anchor[4] == 1:
                        conf_scores.append(anchor[4])
                        x, y, w, h = anchor[:4]
                        y, x = np.asarray([y, x]) * cell_size
                        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
                        classes.append(np.argmax(anchor[5:], axis=-1))

        return np.asarray(conf_scores), np.asarray(boxes), np.asarray(classes)

    def process_sample(self, label):
        """
            Maps the input image to it's corresponding tensor output

            :return: array of shape C x C x ANCHORS x (tx + ty + tw + th + obj_score + C), where:
             tx, ty are the offsets with respect to the grid cell,
             tw, th are the offsets with respect to the anchor, and
             C is the number of classes
            """
        no_anchors = len(self.anchors)
        output = np.zeros((*self.grid_size, no_anchors, (5 + self.no_classes)))
        cell_size = self.image_size / self.grid_size
        true_boxes = np.zeros((1, 1, 1, self.max_boxes_per_image, 4))
        for box_index, (one_hot_encoding, bbox) in enumerate(zip(label[LabelType.CLASS], label[LabelType.COORDINATES])):
            center = np.asarray([np.mean(bbox[1::2]), np.mean(bbox[0::2])])
            cy, cx = np.asarray(center / cell_size, dtype=int)

            # tx = x_center / cell_size  # (x_center - cx * cell_size) / cell_size + cx
            # ty = y_center / cell_size  # (y_center - cy * cell_size) / cell_size + cy

            ty, tx = center / cell_size

            best_anchor = -1
            best_iou = -1
            for anchor_index, anchor in enumerate(self.anchors):
                anchor = anchor[::-1]  # from wh -> hw
                y_min, x_min = center - anchor / 2
                y_max, x_max = center + anchor / 2

                x_min, x_max = np.clip([x_min, x_max], 0, self.image_size[1])
                y_min, y_max = np.clip([y_min, y_max], 0, self.image_size[0])

                current_iou = iou(bbox, [x_min, y_min, x_max, y_max])
                if current_iou > best_iou:
                    best_anchor = anchor_index
                    best_iou = current_iou

            # anchor_width, anchor_height = anchors[best_anchor]
            box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tw = box_width  # np.log(box_width) - np.log(anchor_width)
            th = box_height  # np.log(box_height) - np.log(anchor_height)

            output[cy, cx, best_anchor, 0] = tx
            output[cy, cx, best_anchor, 1] = ty
            output[cy, cx, best_anchor, 2] = tw
            output[cy, cx, best_anchor, 3] = th
            output[cy, cx, best_anchor, 4] = 1.
            output[cy, cx, best_anchor, 5:] = one_hot_encoding

            true_boxes[0, 0, 0, box_index] = [tx, ty, tw, th]
        return output, true_boxes

    def __call__(self, labels):
        encodings, true_bboxes = [], []
        for label in labels:
            encodings, true_boxes = self.process_sample(label)
        return np.asarray(encodings), np.asarray(true_bboxes)


class YOLOPostprocessing:
    def __init__(self, image_size, grid_size, anchors, apply_argmax=True):
        self.image_size = image_size
        self.cell_grid = create_cell_grid(grid_size, len(anchors))
        cell_size = np.asarray(image_size) / np.asarray(grid_size)
        self.cell_size = cell_size[1], cell_size[0]

        self.anchors = np.asarray(anchors)
        self.apply_argmax = apply_argmax

    def __call__(self, output):
        """
        Extracts from the output scores, boxes, and classes

        :param output: prediction made by model
        :return: scores, boxes, classes
        """
        xy = (K.sigmoid(output[..., :2]) + self.cell_grid) * self.cell_size
        wh = K.exp(output[..., 2:4]) * self.anchors
        conf_scores = K.sigmoid(output[..., 4:5])
        classes = conf_scores * softmax(output[..., 5:])

        if self.apply_argmax:
            conf_scores = K.max(classes, axis=-1)
            classes = K.argmax(classes)
        else:
            tf.squeeze(conf_scores)
        return conf_scores, \
               K.clip(tf.concat([xy - wh / 2, xy + wh / 2], axis=-1), min_value=0, max_value=self.image_size[::-1]), \
               classes


class YOLOHead:
    def __init__(self, grid_size, no_anchors, no_classes, conv_generator):
        self.grid_size = grid_size
        self.no_anchors = no_anchors
        self.no_classes = no_classes
        self.conv_1x1 = conv_generator(3, no_anchors * (4 + 1 + no_classes))()

    def __call__(self, inputs):
        x = self.conv_1x1(inputs)
        return Reshape((*self.grid_size, self.no_anchors, 4 + 1 + self.no_classes))(x)


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
    preprocessor = YOLOPreprocessing(
        (10, 20),
        (2, 4),
        np.asarray([
            [2, 2],
            [3, 3]
        ]),
        2,
        5
    )

    processed_gt, true_bboxes = preprocessor(fake_gt)[0]
    print(processed_gt.shape)
    for i in range(processed_gt.shape[0]):
        for j in range(processed_gt.shape[1]):
            print(i, j)
            print(processed_gt[i, j, :, :])
            print()

    conf, boxes, classes = preprocessor.decode_gt(processed_gt)

    print(boxes)
    print(classes)
