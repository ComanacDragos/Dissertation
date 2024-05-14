import numpy as np

from backend.callbacks.box_image_plotter import draw_boxes
from backend.enums import DataType, LabelType, ObjectDetectionOutputType, OutputType


class ObjectDetectionService:
    def __init__(self, data_generator, model=None):
        self.data_generator = data_generator
        self.model = model

        self.names = list(data_generator.data.loc[:, "image"])

    def __getitem__(self, index):
        data = self.data_generator[index]
        image = data[DataType.IMAGE][0]

        gt_img = None
        if DataType.LABEL in data:
            label = data[DataType.LABEL][0]
            if len(label[LabelType.CLASS]) == 0:
                gt_img = image
            else:
                classes = [self.data_generator.labels[x] for x in np.argmax(label[LabelType.CLASS], axis=-1)]
                coordinates = [[int(x) for x in box] for box in label[LabelType.COORDINATES].tolist()]
                gt_img = draw_boxes(image, classes, coordinates)

        predicted_img = None
        if self.model:
            outputs = self.model(image[None, ...])[ObjectDetectionOutputType.AFTER_FILTERING]
            boxes = np.asarray(outputs[OutputType.COORDINATES][0], dtype=int)
            pred_texts = [f"{self.data_generator.labels[int(cls)]}: {str(round(prob, 2))}" for cls, prob in
                          zip(outputs[OutputType.CLASS_LABEL][0], outputs[OutputType.CLASS_PROBABILITIES][0])]

            predicted_img = draw_boxes(image, pred_texts, boxes)

        if gt_img is not None and predicted_img is not None:
            image = np.concatenate([gt_img, np.zeros((20, *gt_img.shape[1:])), predicted_img])
            image = np.array(image, dtype=np.uint8)
        elif gt_img is None:
            image = predicted_img
        else:
            image = gt_img

        return image
