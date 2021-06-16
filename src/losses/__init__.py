from abc import abstractmethod
import numpy as np
from numpy import ndarray


class LossBase:
    def __call__(self, y_true: ndarray, y_pred: ndarray):
        return self.forward(y_true, y_pred)

    @abstractmethod
    def forward(self, y_true: ndarray, y_pred: ndarray):
        raise NotImplementedError

    @abstractmethod
    def grad(self, y_true: ndarray, y_pred: ndarray, x: ndarray, y_grad_fn):
        raise NotImplementedError


class MSELoss(LossBase):
    def __str__(self):
        return "Mean Square Error"

    def forward(self, y_true: ndarray, y_pred: ndarray):
        assert len(y_true) == len(y_pred)
        outputs = (y_true - y_pred) ** 2
        return np.mean(outputs)

    def grad(self, y_true: ndarray, y_pred: ndarray, x: ndarray, y_layer):
        return (y_pred - y_true) * y_layer.grad(x) * (2 / len(y_true))


class CrossEntropy(LossBase):
    def __str__(self):
        return "CrossEntropy"

    def __call__(self, y_true: ndarray, y_pred: ndarray):
        return self.forward(y_true, y_pred)

    def forward(self, y_true: ndarray, y_pred: ndarray):
        eps = np.finfo(float).eps
        return -np.sum(y_true * np.log(y_pred + eps), axis=1)

    def grad(self, y_true: ndarray, y_pred: ndarray, x: ndarray, y_layer):
        pass


if __name__ == '__main__':
    yt = np.arange(10)
    yp = np.arange(10, 20)
    loss = MSELoss()
    val = loss(yt, yp)
    print(val)
