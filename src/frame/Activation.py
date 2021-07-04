from abc import abstractmethod

from numpy import ndarray

from frame.HyperParametersHolder import HyperParametersHolder


class Activation(HyperParametersHolder):
    def __init__(self, **kwargs):
        HyperParametersHolder.__init__(self, **kwargs)

    def __str__(self):
        return Activation.__name__

    def __call__(self, x: ndarray) -> ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        """前向传播"""
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: ndarray) -> ndarray:
        """计算梯度"""
        raise NotImplementedError

    @abstractmethod
    def grad2(self, x: ndarray) -> ndarray:
        """计算梯度的梯度"""
        raise NotImplementedError
