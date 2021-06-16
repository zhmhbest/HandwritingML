from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, x: ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ndarray):
        """前向传播"""
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: ndarray):
        """计算梯度"""
        raise NotImplementedError

    @abstractmethod
    def grad2(self, x: ndarray):
        """计算梯度的梯度"""
        raise NotImplementedError


class Sigmoid(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def forward(self, x: ndarray):
        return 1 / (1 + np.exp(-x))

    def grad(self, x: ndarray):
        fn_x = self.forward(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x: ndarray):
        fn_x = self.forward(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class Tanh(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh"

    def forward(self, x: ndarray):
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return np.tanh(x)

    def grad(self, x: ndarray):
        fn_x = self.forward(x)
        return 1 - fn_x ** 2

    def grad2(self, x: ndarray):
        fn_x = self.forward(x)
        return 2 * (fn_x ** 3 - fn_x)


class ReLU(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def forward(self, x: ndarray):
        return np.maximum(x, 0)
        # return np.clip(x, 0, np.inf)

    def grad(self, x: ndarray):
        return np.where(x > 0, 1, 0)

    def grad2(self, x: ndarray):
        return np.zeros_like(x)


class LeakyReLU(ActivationBase):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def __str__(self):
        return f"Leaky ReLU(alpha={self.alpha})"

    def forward(self, x: ndarray):
        return np.where(x > 0, x, x * self.alpha)

    def grad(self, x: ndarray):
        return np.where(x > 0, 1, self.alpha)

    def grad2(self, x: ndarray):
        return np.zeros_like(x)


class ELU(ActivationBase):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def __str__(self):
        return f"ELU(alpha={self.alpha})"

    def forward(self, x: ndarray):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def grad(self, x: ndarray):
        return np.where(x > 0, 1, self.alpha * np.exp(x))

    def grad2(self, x: ndarray):
        return np.where(x > 0, 0, self.alpha * np.exp(x))


class SoftPlus(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SoftPlus"

    def forward(self, x: ndarray):
        return np.log(1 + np.exp(x))

    def grad(self, x):
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)

    def grad2(self, x):
        exp_x = np.exp(x)
        return exp_x / ((1 + exp_x) ** 2)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes

    def subplot(ax: Axes, x: ndarray, module: ActivationBase):
        ax.plot(inputs, module(x), linestyle='-', label=r"$f(x)$")
        ax.plot(inputs, module.grad(inputs), linestyle='--', label=r"$\dfrac{dy}{dx}$")
        ax.plot(inputs, module.grad2(inputs), linestyle=':', label=r"$\dfrac{d^2y}{dx^2}$")
        ax.grid()
        ax.set_title(str(module))
        ax.legend()

    fig, axs = plt.subplots(2, 3, figsize=[12.8, 2.4])
    inputs = np.linspace(-10, 10).reshape(-1, 1)
    subplot(axs[0][0], inputs, Sigmoid())
    subplot(axs[0][1], inputs, Tanh())
    subplot(axs[0][2], inputs, ReLU())
    subplot(axs[1][0], inputs, LeakyReLU())
    subplot(axs[1][1], inputs, ELU())
    subplot(axs[1][2], inputs, SoftPlus())
    plt.show()
