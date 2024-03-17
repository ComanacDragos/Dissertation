from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import tensorflow as tf

from backend.enums import Stage, DataType
from backend.utils import logger


class GenericDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, root, csv_path, batch_size, stage: Stage, shuffle=True, augmentations=None):
        self.root = Path(root)
        self.data = pd.read_csv(csv_path)
        if not stage == Stage.ALL:
            self.data = self.data[self.data['stage'] == stage.value]
        logger.log(f"Loaded {csv_path} for {stage.value} - {len(self.data)} samples")
        self.batch_size = batch_size
        self.stage = stage
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.ceil(len(self.data) / self.batch_size))

    @abstractmethod
    def create_batch(self, batch_data):
        """
        Creates the batch from the batch_data.
        """

    def __getitem__(self, index) -> Dict[DataType, Any]:
        if index >= self.__len__():
            raise IndexError("Too large index for dataset")
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data = self.data.loc[self.data.index[batch_indices]]
        batch_data = list(row for _, row in batch_data.iterrows())

        return self.create_batch(batch_data)

    def on_epoch_end(self):
        """"
        Called at the end of each epoch
        """
        # if required, shuffle your data after each epoch
        if self.stage == Stage.TRAIN and self.shuffle:
            self.indices = np.arange(len(self.data))
            np.random.shuffle(self.indices)

