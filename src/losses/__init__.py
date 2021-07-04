import numpy as np
from numpy import ndarray

from frame.Loss import Loss


class MSELoss(Loss):
    def __str__(self):
        return "Mean Square Error"

    def loss(self, y_true: ndarray, y_pred: ndarray):
        assert len(y_true) == len(y_pred)
        outputs = (y_true - y_pred) ** 2
        return np.mean(outputs)

    def grad(self, y_true: ndarray, y_pred: ndarray, **kwargs) -> ndarray:
        grad = y_pred - y_true
        return grad


class CrossEntropy(Loss):
    def __str__(self):
        return "CrossEntropy"

    def loss(self, y_true: ndarray, y_pred: ndarray):
        eps = np.finfo(float).eps
        return -np.sum(y_true * np.log(y_pred + eps), axis=1)

    def grad(self, y_true: ndarray, y_pred: ndarray, **kwargs):
        # is_binary(y)
        # is_stochastic(y_pred)
        grad = y_pred - y_true
        return grad


if __name__ == '__main__':
    yt = np.arange(10)
    yp = np.arange(10, 20)
    loss = MSELoss()
    val = loss(yt, yp)
    print(val)
