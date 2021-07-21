from abc import abstractmethod

from numpy import ndarray


class Layer:
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    def __call__(self, x: ndarray) -> ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        """前向传播"""
        raise NotImplementedError()


class FunctionLayer(Layer):
    """
    无参数的Layer
    """
    @abstractmethod
    def grad(self, x: ndarray, **kwargs) -> ndarray:
        """计算梯度"""
        raise NotImplementedError()


class ActivationLayer(FunctionLayer):
    """
    激活函数
    """
    @abstractmethod
    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        """计算梯度的梯度"""
        raise NotImplementedError()


class ParameterLayer:
    """
    有参数的Layer
    """
    @abstractmethod
    def backward(self, pl_py: ndarray):
        """反向传播"""
        raise NotImplementedError()
