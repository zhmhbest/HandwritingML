from abc import abstractmethod

from numpy import ndarray

from frame.Freezable import Freezable
from frame.Optimizer import Optimizer
from frame.ParametersHolder import ParametersHolder


class Layer(Freezable, ParametersHolder):
    def __init__(self, **kwargs):
        Freezable.__init__(self)
        ParametersHolder.__init__(self, **kwargs)

    def __str__(self):
        return Layer.__name__

    def __call__(self, x: ndarray, **kwargs) -> ndarray:
        return self.forward(x, **kwargs)

    def zero_gradients(self):
        """重置梯度"""
        if not self.is_freeze:
            ParametersHolder._zero_gradients(self)

    def update_gradients(self, optimizer: Optimizer, loss):
        """更新梯度"""
        if not self.is_freeze:
            ParametersHolder._update_gradients(self, optimizer, loss)

    @abstractmethod
    def forward(self, x: ndarray, **kwargs) -> ndarray:
        """前向传播"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, out: ndarray, **kwargs):
        """反向传播"""
        raise NotImplementedError
