import numpy as np
from numpy import ndarray

from hwml.frame.loss import Loss


class MSELoss(Loss):
    def loss(self, y_true: ndarray, y_pred: ndarray) -> float:
        return 0.5 * np.square(np.linalg.norm(y_pred - y_true, ord=2))

    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return y_pred - y_true


class CrossEntropy(Loss):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = np.finfo(float).eps if eps is None else eps

    def loss(self, y_true: ndarray, y_pred: ndarray) -> float:
        return -np.sum(y_true * np.log(y_pred + self.eps))

    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return y_pred - y_true


if __name__ == '__main__':
    BATCH_SIZE = 2

    def test1():
        y1 = np.random.randn(BATCH_SIZE, 3)
        y2 = np.random.randn(BATCH_SIZE, 3)
        loss = MSELoss()
        print(loss(y1, y2))
        print(loss.gradients)

    def test2():
        def get_y():
            y = 1 / (1 + np.exp(-np.random.randn(BATCH_SIZE, 1)))
            y = np.hstack([y, np.ones_like(y) - y])
            return y

        y1 = get_y()
        y2 = get_y()
        loss = CrossEntropy()
        print(loss(y1, y2))
        print(loss.gradients)

    test1()
    test2()
