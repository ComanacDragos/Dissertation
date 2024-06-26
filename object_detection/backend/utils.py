import json
import os
import pickle
import random
import threading

import numpy as np
import tensorflow as tf


def set_seed(seed):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def to_json(obj, file):
    json_obj = json.dumps(obj, indent=4, cls=NumpyEncoder)
    with open(file, "w") as f:
        f.write(json_obj)


def open_json(file):
    with open(file) as f:
        return json.load(f)


def to_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def open_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def sum_over_axes(input_tensor, axes):
    """
    Sums a tensor along multiple axes

    :param input_tensor: multi dimensional input tensor
    :param axes: one or more consecutive axis on which the sum is performed
    :return: the reduced tensor
    """
    x = input_tensor
    for a in reversed(axes):
        x = tf.reduce_sum(x, axis=a)
    return x


def my_round(a, threshold=0.5):
    """
    If the difference between the input float and the nearest integer smaller than the float is smaller than the
    threshold, then this integer is returned, otherwise the next integer is returned

    :param a: input float
    :param threshold: rounding threshold, less than 1
    :return: rounded float
    """
    return int(a) if a - int(a) < threshold else int(a) + 1


def run_task(items, target, args, cpu_count=os.cpu_count()):
    """
    Splits items in several batches and runs target function with args on each batch in parallel
    (on same number of threads as cpu cores)

    :param items: list of items to be processed in parallel
    :param target: worker function (first parameter must be a list of items to be processed)
    :param args: the parameters that the worker function is called with
    :param cpu_count: number of cores to be used
    """
    chunk = len(items) // cpu_count
    threads = []
    for startIndex in range(0, cpu_count):
        if startIndex == cpu_count - 1:
            filesToProcess = items[startIndex * chunk:]
        else:
            filesToProcess = items[startIndex * chunk: (startIndex + 1) * chunk]
        thread = threading.Thread(target=target, args=[filesToProcess] + args)
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()


def with_bounding_boxes(img, bounding_boxes, width=3):
    """
    :param img: RGB image as an array
    :param bounding_boxes: list of bounding boxes
    :param width: the width of each bounding box
    :return: the input image but with the bounding boxes drawn over it
    """
    copy = np.copy(img)
    for bbox in bounding_boxes:
        color = [0, 0, 0]
        color[bbox.c % 3] = 255
        copy[bbox.y_min - width:bbox.y_min + width, bbox.x_min:bbox.x_max] = color
        copy[bbox.y_max - width:bbox.y_max + width, bbox.x_min:bbox.x_max] = color
        copy[bbox.y_min:bbox.y_max, bbox.x_min - width:bbox.x_min + width] = color
        copy[bbox.y_min:bbox.y_max, bbox.x_max - width:bbox.x_max + width] = color
    return copy


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def iou(bbox, other_bbox):
    """
    :param bbox: array of coordinates x_min, y_min, x_max, y_max
    :param other_bbox: array of coordinates x_min, y_min, x_max, y_max
    :return: Intersection/Union
    """
    x_min, y_min, x_max, y_max = bbox
    other_x_min, other_y_min, other_x_max, other_y_max = other_bbox

    intersect_width = min(x_max, other_x_max) - max(x_min, other_x_min)
    intersect_height = min(y_max, other_y_max) - max(y_min, other_y_min)

    if intersect_width <= 0 or intersect_height <= 0:
        return 0

    bbox_width, bbox_height = x_max - x_min, y_max - y_min
    other_bbox_width, other_bbox_height = other_x_max - other_x_min, other_y_max - other_y_min

    intersect = intersect_height * intersect_width
    union = bbox_height * bbox_width + other_bbox_height * other_bbox_width - intersect
    return float(intersect) / union
