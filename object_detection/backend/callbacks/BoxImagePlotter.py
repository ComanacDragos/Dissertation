import os
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.callbacks import Callback

from backend.trainer.state import TrainState, EvalState
from backend.enums import DataType, LabelType, OutputType


def draw_boxes(img, text_labels, boxes):
    img = np.copy(img)
    img_width = img.shape[1]
    img_height = img.shape[0]
    for text, box in zip(text_labels, boxes):
        xmin = max(box[0], 0)
        ymin = max(box[1], 0)
        xmax = min(box[0] + box[2], img_width)
        ymax = min(box[1] + box[3], img_height)

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

    def on_train_batch_end(self, batch, logs: TrainState = None):
        if (batch + 1) % self.plot_frequency != 0:
            return
        self._plot_images(self.output_path / "train", batch, logs)

    def on_test_batch_end(self, batch, logs: EvalState = None):
        if (batch + 1) % self.plot_frequency != 0:
            return
        self._plot_images(self.output_path / "eval", batch, logs)

    def on_predict_batch_end(self, batch, logs: EvalState = None):
        if (batch + 1) % self.plot_frequency != 0:
            return
        self._plot_images(self.output_path / "predict", batch, logs)

    def _plot_images(self, path, batch, logs):
        path /= f"epoch_{logs.epoch}_step_{batch}"
        os.makedirs(path, exist_ok=True)
        identifiers = logs.inputs[DataType.IDENTIFIER]
        images = logs.inputs[DataType.IMAGE]
        # TODO continue ...
        #
        # labels = logs.inputs[DataType.LABEL]
        # gt_classes, gt_boxes = labels[LabelType.CLASS], labels[LabelType.COORDINATES]
        #
        # gt_texts = [self.id_to_class[cls] for cls in gt_classes]
        #
        # # gt_images = [draw_boxes() for img, text, boxes]
        #
        # outputs = logs.predictions
        # pred_boxes = outputs[OutputType.COORDINATES]
        # pred_classes = outputs[OutputType.CLASS_LABEL]
        # pred_probs = outputs[OutputType.CLASS_PROBABILITIES]
        #

