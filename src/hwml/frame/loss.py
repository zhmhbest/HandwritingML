from abc import abstractmethod
from numpy import ndarray


class Loss:
    def __init__(self):
        # 在计算损失后保留梯度
        self.gradients = []

    def __call__(self, y_true: ndarray, y_pred: ndarray) -> float:
        v = self.loss(y_true, y_pred)
        g = self.grad(y_true, y_pred)
        self.gradients.append(g)
        return v

    def clear_gradients(self):
        self.gradients.clear()

    @abstractmethod
    def loss(self, y_true: ndarray, y_pred: ndarray) -> float:
        raise NotImplementedError()

    @abstractmethod
    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        """损失对output的梯度，shape与output一致"""
        raise NotImplementedError()
