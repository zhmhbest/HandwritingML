from abc import ABC


class BaseModule(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, **kwargs):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """前向传播"""
        raise NotImplementedError

    def backward(self, **kwargs):
        """反向传播"""
        pass

    def grad(self, **kwargs):
        """计算梯度"""
        raise NotImplementedError

    def grad2(self, **kwargs):
        """计算梯度的梯度"""
        pass
