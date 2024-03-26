import os
from pathlib import Path

import cv2
import numpy as np
from overrides import overrides
from tensorflow.keras.callbacks import Callback

from backend.enums import DataType, OutputType, LabelType, ObjectDetectionOutputType
from backend.trainer.state import TrainState, EvalState


def draw_boxes(img, text_labels, boxes):
    img = np.copy(img)
    img_width = img.shape[1]
    img_height = img.shape[0]
    for text, box in zip(text_labels, boxes):
        xmin = max(box[0], 0)
        ymin = max(box[1], 0)
        xmax = min(box[2], img_width)
        ymax = min(box[3], img_height)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if ymax + 30 >= img_height:
            cv2.rectangle(img, (xmin, ymin),
                          (xmin + len(text) * 10, int(ymin - 20)),
                          (255, 140, 0), cv2.FILLED)
            cv2.putText(img, text, (xmin, int(ymin - 5)), font,
                        0.5, (255, 255, 255), 1)
        else:
            cv2.rectangle(img, (xmin, ymax),
                          (xmin + len(text) * 10, int(ymax + 20)),
                          (255, 140, 0), cv2.FILLED)
            cv2.putText(img, text, (xmin, int(ymax + 15)), font,
                        0.5, (255, 255, 255), 1)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 0, 255), 1)

    return img


class BoxImagePlotter(Callback):
    def __init__(self, output_path, plot_frequency, labels):
        super().__init__()
        self.plot_frequency = plot_frequency
        self.id_to_class = {i: label for i, label in enumerate(labels)}
        self.output_path = Path(output_path) / "image_plots"

    @overrides
    def on_train_batch_end(self, batch, logs: TrainState = None):
        if (batch + 1) % self.plot_frequency != 0:
            return
        self._plot_images(self.output_path / "train", batch, logs)

    @overrides
    def on_test_batch_end(self, batch, logs: EvalState = None):
        if (batch + 1) % self.plot_frequency != 0:
            return
        self._plot_images(self.output_path / "eval", batch, logs)

    @overrides
    def on_predict_batch_end(self, batch, logs: EvalState = None):
        if (batch + 1) % self.plot_frequency != 0:
            return
        self._plot_images(self.output_path / "predict", batch, logs)

    def _plot_images(self, path, batch, logs):
        path /= f"epoch_{logs.epoch}_step_{batch}"
        os.makedirs(path, exist_ok=True)
        identifiers = [Path(identifier).name for identifier in logs.inputs[DataType.IDENTIFIER]]
        images = logs.inputs[DataType.IMAGE]

        labels = logs.inputs[DataType.LABEL]

        gt_texts = [
            [self.id_to_class[np.argmax(one_hot_encoding, axis=-1)] for one_hot_encoding in label[LabelType.CLASS]]
            for label in labels
        ]

        gt_boxes = [np.asarray(label[LabelType.COORDINATES], dtype=int) for label in labels]

        gt_images = [draw_boxes(img, text, boxes)
                     for img, text, boxes in zip(images, gt_texts, gt_boxes)
                     ]

        if ObjectDetectionOutputType.AFTER_FILTERING in logs.predictions:
            outputs = logs.predictions[ObjectDetectionOutputType.AFTER_FILTERING]
        else:
            outputs = logs.predictions[ObjectDetectionOutputType.BEFORE_FILTERING]
        pred_boxes = outputs[OutputType.COORDINATES]
        pred_classes = outputs[OutputType.CLASS_LABEL]
        pred_probs = outputs[OutputType.CLASS_PROBABILITIES]

        pred_texts = [
            [f"{self.id_to_class[cls]}: {str(round(prob, 2))}" for cls, prob in zip(pred_classes_img, pred_probs_img)]
            for pred_classes_img, pred_probs_img in zip(pred_classes, pred_probs)
        ]

        pred_images = [draw_boxes(img, text, np.asarray(boxes, dtype=int))
                       for img, text, boxes in zip(images, pred_texts, pred_boxes)
                       ]

        final_images = [
            np.concatenate([gt_image, np.zeros((20, *gt_image.shape[1:]), dtype=int), pred_image])
            for gt_image, pred_image in zip(gt_images, pred_images)
        ]

        for identifier, image in zip(identifiers, final_images):
            cv2.imwrite(str(path / identifier), image)
