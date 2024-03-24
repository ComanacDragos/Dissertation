class TrainState:
    def __init__(self, epoch, inputs=None, predictions=None, optimizer=None, model=None, loss=None):
        self.epoch = epoch
        self.inputs = inputs
        self.optimizer = optimizer
        self.model = model
        self.loss = loss
        self.predictions= predictions


class EvalState:
    def __init__(self, epoch, inputs=None, predictions=None, model=None, loss=None):
        self.inputs = inputs
        self.epoch = epoch
        self.predictions = predictions
        self.model = model
        self.loss = loss
