from abc import abstractmethod

import cv2
import numpy as np
import tensorflow as tf
from overrides import overrides

from backend.enums import Stage, DataType
from backend.logger import logger


class MNISTDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, stage: Stage, shuffle=True, augmentations=None):
        data = tf.keras.datasets.mnist.load_data()
        if stage == Stage.TRAIN:
            self.images, self.labels = data[0]
        else:
            self.images, self.labels = data[1]

        self.images = tf.keras.utils.normalize(self.images)
        self.images = tf.expand_dims(self.images, axis=-1)

        self.batch_size = batch_size
        logger.log(
            f"Loaded {stage.value} - {len(self.images)} samples - bs: {batch_size} - len: {self.__len__()}")

        self.stage = stage
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.indices = np.arange(len(self.images))
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError("Too large index for dataset")
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        return {
            DataType.IDENTIFIER: batch_indices,
            DataType.IMAGE: tf.gather(self.images, batch_indices),
            DataType.LABEL: tf.gather(self.labels, batch_indices)
        }

    def on_epoch_end(self):
        """"
        Called at the end of each epoch
        """
        # if required, shuffle your data after each epoch
        if self.stage == Stage.TRAIN and self.shuffle:
            self.indices = np.arange(len(self.images))
            np.random.shuffle(self.indices)
