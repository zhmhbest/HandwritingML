from typing import Tuple

import numpy as np
from numpy import ndarray


class Parameter(object):
    @staticmethod
    def he_uniform(shape: Tuple[int, int]):
        dim_i, dim_o = shape
        b = np.sqrt(6 / dim_i)
        return np.random.uniform(-b, b, size=shape)

    def __init__(self, shape: Tuple[int, int], initializer: str):
        if 'zero' == initializer:
            param = np.zeros(shape=shape)
        elif 'he_uniform' == initializer:
            param = self.he_uniform(shape)
        else:
            raise ValueError("Unrecognized initializer.")
        self.param: ndarray = param
        self.grad: ndarray = np.zeros_like(param)

    def zero_grad(self):
        self.grad = np.zeros_like(self.param)

    def accumulation_grad(self, grad):
        self.grad += grad

    def update_param(self, fn):
        self.param = fn(self.param, self.grad)

    def update(self, fn):
        self.update_param(fn)
        self.zero_grad()


class Linear(object):
    def __init__(self, input_dim: int, output_dim: int):
        self.io_shape = (input_dim, output_dim)
        # 参数
        self.w = Parameter(self.io_shape, 'he_uniform')
        self.b = Parameter((1, output_dim), 'zero')
        # 记忆Forward时的变量
        self.derivative_x = None

    def __call__(self, x: ndarray):
        return self.forward(x)

    def forward(self, x: ndarray):
        """
        .. math::
            z = xw + b
        """
        self.derivative_x = x
        z = x @ self.w.param + self.b.param
        return z

    def backward(self, pl_pz: ndarray):
        """
        .. math::
            \dfrac{∂L}{∂w} = \dfrac{∂L}{∂z} \dfrac{∂z}{∂w} = x^T \dfrac{∂L}{∂z}

            \dfrac{∂L}{∂b} = \dfrac{∂L}{∂z} \dfrac{∂z}{∂b} = \dfrac{∂L}{∂z}
        """
        x = self.derivative_x
        self.w.accumulation_grad(x.T @ pl_pz)
        self.b.accumulation_grad(np.sum(pl_pz, axis=0, keepdims=True))

    def update(self, lr: float = 0.01):
        update = (lambda w, g: w - lr * g)
        self.w.update(update)
        self.b.update(update)


class MSELoss(object):
    @staticmethod
    def loss(y_true: ndarray, y_pred: ndarray) -> float:
        # return 0.5 * np.mean(np.square(y_pred - y_true), axis=-1)
        return 0.5 * np.square(np.linalg.norm(y_pred - y_true, ord=2))

    @staticmethod
    def grad(y_true: ndarray, y_pred: ndarray) -> ndarray:
        return y_pred - y_true

    def __call__(self, y_true: ndarray, y_pred: ndarray) -> Tuple[float, ndarray]:
        return MSELoss.loss(y_true, y_pred), MSELoss.grad(y_true, y_pred)


if __name__ == '__main__':
    loss = MSELoss()
    linear = Linear(2, 1)

    _x = np.random.rand(1, 2)
    _y = np.sum(_x, axis=-1, keepdims=True) * 3

    for i in range(10):
        _p = linear(_x)
        _l, _g = loss(_y, _p)
        print(_l)

        linear.backward(_g)
        linear.update()

