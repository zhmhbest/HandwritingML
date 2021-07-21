from typing import Union

from numpy import ndarray
from hwml.frame import Model, Layer, Loss, Optimizer, ParameterLayer


class Sequential(Model):
    def __init__(self):
        self.layers = []
        self.loss: Union[None, Loss] = None
        self.optimizer: Union[None, Optimizer] = None

    def add(self, layer: Layer):
        self.layers.append(layer)

    def set_loss(self, loss: Loss):
        self.loss = loss

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        for layer in self.layers:
            if isinstance(layer, ParameterLayer):
                layer.optimizer = optimizer

    def __call__(self, x: ndarray):
        return self.forward(x)

    def forward(self, x: ndarray):
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, x_train: ndarray, y_train: ndarray):
        assert self.optimizer is not None
        assert self.loss is not None

        self.optimizer.step()

        # forward
        y_pred = self.forward(x_train)
        loss_val = self.loss(y_train, y_pred)

        # backward
        g = self.loss.gradients
        for i in range(len(self.layers) - 1, -1, -1):
            layer: ParameterLayer = self.layers[i]
            g = layer.backward(g)

        self.loss.clear_gradients()
        return loss_val
