from abc import abstractmethod
from typing import Union

import numpy as np
from numpy import ndarray


class ActivationBase:
    def __init__(self, **kwargs):
        # 超参数设置
        self.hyper_parameters = kwargs

    def __str__(self):
        return "Activation"

    def __call__(self, x: ndarray):
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
    def __init__(self, alpha: float = 0.3):
        assert isinstance(alpha, float), "Unsupported alpha type"
        super().__init__(alpha=alpha)

    def __str__(self):
        return f"Leaky ReLU(alpha={self.hyper_parameters['alpha']})"

    def forward(self, x: ndarray):
        return np.where(x > 0, x, x * self.hyper_parameters['alpha'])

    def grad(self, x: ndarray):
        return np.where(x > 0, 1, self.hyper_parameters['alpha'])

    def grad2(self, x: ndarray):
        return np.zeros_like(x)


class ELU(ActivationBase):
    def __init__(self, alpha=1.0):
        assert isinstance(alpha, float), "Unsupported alpha type"
        super().__init__(alpha=alpha)

    def __str__(self):
        return f"ELU(alpha={self.hyper_parameters['alpha']})"

    def forward(self, x: ndarray):
        return np.where(x > 0, x, self.hyper_parameters['alpha'] * (np.exp(x) - 1))

    def grad(self, x: ndarray):
        return np.where(x > 0, 1, self.hyper_parameters['alpha'] * np.exp(x))

    def grad2(self, x: ndarray):
        return np.where(x > 0, 0, self.hyper_parameters['alpha'] * np.exp(x))


class SoftPlus(ActivationBase):
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


class ActivationInitializer:
    def __init__(self, name: Union[str, ActivationBase] = None, **kwargs):
        if name is None:
            # 默认激活函数
            self.activation_class = Tanh
        elif isinstance(name, str):
            # 字符串
            activation_initializer_string_names = {
                'sigmoid': Sigmoid,
                'tanh': Tanh,
                'relu': ReLU,
                'leaky relu': LeakyReLU,
                'leaky_relu': LeakyReLU,
                'elu': ELU,
                'soft plus': SoftPlus,
                'soft_plus': SoftPlus
            }
            name = name.lower()
            if name not in activation_initializer_string_names.keys():
                raise ValueError(f"Unrecognized activation: `{name}`")
            self.activation_class = activation_initializer_string_names[name]
        elif isinstance(name, ActivationBase):
            self.activation_class = name
        else:
            raise ValueError(f"Unrecognized activation name type")
        # 保存超参数
        self.hyper_parameters = kwargs

    def __call__(self) -> ActivationBase:
        # 创建一个激活函数实列
        return self.activation_class(**self.hyper_parameters)


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

    fig, axs = plt.subplots(2, 3, figsize=[16, 9], dpi=100)
    inputs = np.linspace(-10, 10).reshape(-1, 1)
    subplot(axs[0][0], inputs, Sigmoid())
    subplot(axs[0][1], inputs, Tanh())
    subplot(axs[0][2], inputs, ReLU())
    subplot(axs[1][0], inputs, LeakyReLU())
    subplot(axs[1][1], inputs, ELU())
    subplot(axs[1][2], inputs, SoftPlus())
    # plt.savefig("./img.png")
    plt.show()
