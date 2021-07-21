from abc import abstractmethod
from typing import Union

from numpy import ndarray

from .loss import Loss
from .optimizer import Optimizer
from .layer import Layer, ParameterLayer


class Model:
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

    @abstractmethod
    def train(self, x_train: ndarray, y_train: ndarray):
        raise NotImplementedError()
