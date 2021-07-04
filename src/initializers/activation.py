import re

import numpy as np
from numpy import ndarray

from frame.Activation import Activation


class Sigmoid(Activation):
    def __str__(self):
        return Sigmoid.__name__

    def forward(self, x: ndarray):
        return 1 / (1 + np.exp(-x))

    def grad(self, x: ndarray):
        fn_x = self.forward(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x: ndarray):
        fn_x = self.forward(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class Tanh(Activation):
    def __str__(self):
        return Tanh.__name__

    def forward(self, x: ndarray):
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return np.tanh(x)

    def grad(self, x: ndarray):
        fn_x = self.forward(x)
        return 1 - fn_x ** 2

    def grad2(self, x: ndarray):
        fn_x = self.forward(x)
        return 2 * (fn_x ** 3 - fn_x)


class ReLU(Activation):
    def __str__(self):
        return ReLU.__name__

    def forward(self, x: ndarray):
        return np.maximum(x, 0)
        # return np.clip(x, 0, np.inf)

    def grad(self, x: ndarray):
        return np.where(x > 0, 1, 0)

    def grad2(self, x: ndarray):
        return np.zeros_like(x)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.3):
        if isinstance(alpha, str):
            alpha = float(alpha)
        assert isinstance(alpha, float), "Unsupported alpha type"
        super().__init__(alpha=alpha)

    def __str__(self):
        return f"{LeakyReLU.__name__}(alpha={self.hyper_parameters['alpha']})"

    def forward(self, x: ndarray):
        return np.where(x > 0, x, x * self.hyper_parameters['alpha'])

    def grad(self, x: ndarray):
        return np.where(x > 0, 1, self.hyper_parameters['alpha'])

    def grad2(self, x: ndarray):
        return np.zeros_like(x)


class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        if isinstance(alpha, str):
            alpha = float(alpha)
        assert isinstance(alpha, float), "Unsupported alpha type"
        super().__init__(alpha=alpha)

    def __str__(self):
        return f"{ELU.__name__}(alpha={self.hyper_parameters['alpha']})"

    def forward(self, x: ndarray):
        return np.where(x > 0, x, self.hyper_parameters['alpha'] * (np.exp(x) - 1))

    def grad(self, x: ndarray):
        return np.where(x > 0, 1, self.hyper_parameters['alpha'] * np.exp(x))

    def grad2(self, x: ndarray):
        return np.where(x > 0, 0, self.hyper_parameters['alpha'] * np.exp(x))


class SoftPlus(Activation):
    def __str__(self):
        return SoftPlus.__name__

    def forward(self, x: ndarray):
        return np.log(1 + np.exp(x))

    def grad(self, x):
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)

    def grad2(self, x):
        exp_x = np.exp(x)
        return exp_x / ((1 + exp_x) ** 2)


class LinearActivation(Activation):
    """
        用于保持线性
    """
    def __str__(self):
        return LinearActivation.__name__

    def forward(self, x: ndarray):
        return x

    def grad(self, x: ndarray):
        return np.ones_like(x)

    def grad2(self, x: ndarray):
        return np.zeros_like(x)


_initializers = {
    r'^$': LinearActivation,
    r'^sigmoid$': Sigmoid,
    r'^tanh$': Tanh,
    r'^relu$': ReLU,
    rf'^leaky relu\((.+?)=(.+?)\)$': LeakyReLU,
    rf'^leakyrelu\((.+?)=(.+?)\)$': LeakyReLU,
    rf'^elu\((.+?)=(.+?)\)$': ELU,
    r'^soft plus$': SoftPlus,
    r'^softplus$': SoftPlus
}


class ActivationInitializer:
    def __init__(self, activation_name: str):
        self.activation_class, self.hyper_parameters = self.from_str(activation_name)

    @staticmethod
    def from_str(activation_name: str) -> (type, dict):
        activation_name = activation_name.lower()
        activation = None
        params = {}
        groups = None
        for pattern in _initializers.keys():
            r = re.match(pattern, activation_name)
            if r is not None:
                activation = _initializers[pattern]
                groups = r.groups()
                break
        if activation is None:
            raise ValueError(f"Unknown activation: {activation_name}")
        else:
            if len(groups) > 0:
                params = dict(zip(
                    [groups[i] for i in range(0, len(groups), 2)],
                    [groups[i] for i in range(1, len(groups), 2)]
                ))
            return activation, params

    def __call__(self) -> Activation:
        # 创建一个激活函数实列
        return self.activation_class(**self.hyper_parameters)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes

    def subplot(ax: Axes, x: ndarray, module: Activation):
        ax.plot(inputs, module(x), linestyle='-', label=r"$f(x)$")
        ax.plot(inputs, module.grad(inputs), linestyle='--', label=r"$\dfrac{dy}{dx}$")
        ax.plot(inputs, module.grad2(inputs), linestyle=':', label=r"$\dfrac{d^2y}{dx^2}$")
        ax.grid()
        ax.set_title(str(module))
        ax.legend()

    fig, axs = plt.subplots(2, 3, figsize=[16, 9], dpi=100)
    inputs = np.linspace(-10, 10).reshape(-1, 1)
    subplot(axs[0][0], inputs, ActivationInitializer("sigmoid")())
    subplot(axs[0][1], inputs, ActivationInitializer("tanh")())
    subplot(axs[0][2], inputs, ActivationInitializer("relu")())
    subplot(axs[1][0], inputs, ActivationInitializer("LeakyReLU(alpha=0.3)")())
    subplot(axs[1][1], inputs, ActivationInitializer("ELU(alpha=0.5)")())
    subplot(axs[1][2], inputs, ActivationInitializer("SoftPlus")())
    # subplot(axs[2][0], inputs, ActivationInitializer("")())
    # plt.savefig("./img.png")
    plt.show()
