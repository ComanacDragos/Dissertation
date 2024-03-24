import time

import tensorflow as tf
from tensorflow.keras.callbacks import CallbackList
from backend.trainer.state import TrainState, EvalState
from tqdm import tqdm
from backend.utils import logger
from backend.enums import DataType


class GenericTrainer:
    def __init__(
            self,
            train_dataset,
            val_dataset,
            loss,
            optimizer,
            callbacks: CallbackList,
            model,
            preprocessor,
            postprocessor,
            epochs
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss = loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.epochs = epochs

    def train(self):
        self.model.compile(self.optimizer)

        self.callbacks.on_train_begin()
        for epoch in range(self.epochs):
            logger.log(f"\nTrain epoch {epoch}")
            self.callbacks.on_epoch_begin(epoch)
            self.train_loop(epoch)
            logger.log(f"\nEval epoch {epoch}")
            self.eval_loop(epoch)
            self.callbacks.on_epoch_end(epoch)
        self.callbacks.on_train_end()

    def compute_loss(self, inputs, network_output):
        labels = inputs[DataType.LABEL]
        if self.preprocessor:
            labels = self.preprocessor(labels)
        loss = self.loss(labels, network_output)
        return loss

    def forward(self, samples):
        predictions = self.model(samples)
        return predictions

    def train_loop(self, epoch):
        self.callbacks.on_train_begin()
        for step, inputs in (pbar := tqdm(enumerate(self.train_dataset), total=len(self.train_dataset), miniters=0)):
            self.callbacks.on_train_batch_begin(
                step,
                logs=TrainState(
                    epoch=epoch,
                    inputs=inputs,
                    optimizer=self.optimizer,
                    model=self.model,
                )
            )
            samples, labels = inputs[DataType.IMAGE], inputs[DataType.LABEL]

            with tf.GradientTape() as tape:
                network_output = self.forward(samples)
                loss_value = self.compute_loss(inputs, network_output)

            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            loss_value = loss_value.numpy()
            pbar.set_postfix_str(f"Train Loss: {round(loss_value, 4)}")

            if self.postprocessor:
                predictions = self.postprocessor(network_output)
            else:
                predictions = network_output

            self.callbacks.on_train_batch_end(
                step,
                logs=TrainState(
                    epoch=epoch,
                    inputs=inputs,
                    optimizer=self.optimizer,
                    model=self.model,
                    loss=loss_value,
                    predictions=predictions
                )
            )
        self.callbacks.on_train_end()

    def eval_loop(self, epoch):
        self.callbacks.on_test_begin()
        for step, inputs in (pbar := tqdm(enumerate(self.val_dataset), total=len(self.val_dataset), miniters=0)):
            self.callbacks.on_test_batch_begin(
                step,
                EvalState(epoch, inputs=inputs)
            )
            samples, labels = inputs[DataType.IMAGE], inputs[DataType.LABEL]
            network_output = self.forward(samples)

            loss_value = self.compute_loss(inputs, network_output).numpy()
            pbar.set_postfix_str(f"Eval Loss: {round(loss_value, 4)}")

            if self.postprocessor:
                predictions = self.postprocessor(network_output)
            else:
                predictions = network_output

            self.callbacks.on_test_batch_end(
                step,
                EvalState(
                    epoch,
                    inputs=inputs,
                    predictions=predictions,
                    model=self.model,
                    loss=loss_value
                )
            )
        self.callbacks.on_test_end(
            logs=EvalState(
                epoch,
                model=self.model
            )
        )
