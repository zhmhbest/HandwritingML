"""
    这是一个仅实现了Linear层和MSE的简单示例
"""
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
        self.shape = shape
        self.param: ndarray = param
        self.grad: ndarray = np.zeros_like(param)

    def __call__(self) -> ndarray:
        return self.param

    def zero_grad(self):
        self.grad = np.zeros_like(self.param)

    def accumulate_grad(self, grad: ndarray):
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
        z = x @ self.w() + self.b()
        return z

    def backward(self, pl_pz: ndarray):
        """
        .. math::
            \dfrac{∂L}{∂w} = \dfrac{∂L}{∂z} \dfrac{∂z}{∂w} = x^T \dfrac{∂L}{∂z}

            \dfrac{∂L}{∂b} = \dfrac{∂L}{∂z} \dfrac{∂z}{∂b} = \dfrac{∂L}{∂z}
        """
        x = self.derivative_x
        self.w.accumulate_grad(x.T @ pl_pz)
        self.b.accumulate_grad(np.sum(pl_pz, axis=0, keepdims=True))

    def update(self, lr: float = 0.01):
        update = (lambda w, g: w - lr * g)
        self.w.update(update)
        self.b.update(update)


class MSELoss(object):
    @staticmethod
    def loss(y_true: ndarray, y_pred: ndarray) -> float:
        return 0.5 * np.square(np.linalg.norm(y_pred - y_true, ord=2))

    @staticmethod
    def grad(y_true: ndarray, y_pred: ndarray) -> ndarray:
        return y_pred - y_true

    def __call__(self, y_true: ndarray, y_pred: ndarray) -> Tuple[float, ndarray]:
        return MSELoss.loss(y_true, y_pred), MSELoss.grad(y_true, y_pred)


if __name__ == '__main__':
    def main():
        from matplotlib import pyplot as plt
        batch_size = 32
        input_dim = 1
        output_dim = 1

        loss = MSELoss()
        linear = Linear(input_dim, output_dim)

        x_train = np.random.rand(batch_size, input_dim)
        y_train = np.sum(x_train, axis=-1, keepdims=True) * 3
        x_test = np.random.rand(batch_size, input_dim)
        y_test = np.sum(x_test, axis=-1, keepdims=True) * 3

        for i in range(100):
            _p = linear(x_train)
            _l, _g = loss(y_train, _p)
            print(_l)

            linear.backward(_g)
            linear.update()

        y_pred = linear(x_test)
        plt.plot([x_test[i] for i in range(batch_size)], [y_test[i] for i in range(batch_size)])
        plt.plot([x_test[i] for i in range(batch_size)], [y_pred[i] for i in range(batch_size)])
        plt.grid()
        plt.show()
    main()
