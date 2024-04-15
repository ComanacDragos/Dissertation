import numpy as np

from collections import OrderedDict


def compute_confusion_matrix(gt, pred, num_classes=10):
    conf_mat = np.zeros((num_classes, num_classes))
    np.add.at(conf_mat, (pred, gt), 1)
    return conf_mat


class Accuracy:
    def __init__(self, no_classes):
        self.no_classes = no_classes

    def __call__(self, logs):
        predictions = logs['predictions']
        labels = logs['labels']

        conf_mat = np.zeros((self.no_classes, self.no_classes))
        np.add.at(conf_mat, (predictions, labels), 1)

        acc = np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)
        metrics = OrderedDict()
        metrics['acc'] = acc
        print(acc)

        # for i, label in enumerate(self.labels):
        #     metrics[f'{label}_AP'] = aps[i]['ap']

        return metrics
