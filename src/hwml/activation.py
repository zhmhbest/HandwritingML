import numpy as np
from numpy import ndarray
from hwml.frame import ActivationLayer as Activation


class Affine(Activation):
    def __init__(self, slope: float = 1.0, intercept: float = 0.0):
        self.slope = slope
        self.intercept = intercept

    def __str__(self) -> str:
        return f"{Affine.__name__}(slope={self.slope}, intercept={self.intercept})"

    def forward(self, x: ndarray) -> ndarray:
        return self.slope * x + self.intercept

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return self.slope * np.ones_like(x)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.zeros_like(x)


class Sigmoid(Activation):
    def __str__(self) -> str:
        return Sigmoid.__name__

    def forward(self, x: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-x))

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        f = kwargs['f'] if 'f' in kwargs else self.forward(x)
        return f * (1 - f)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        f = kwargs['f'] if 'f' in kwargs else self.forward(x)
        return f * (1 - f) * (1 - 2 * f)


class Tanh(Activation):
    def __str__(self) -> str:
        return Tanh.__name__

    def forward(self, x: ndarray) -> ndarray:
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return np.tanh(x)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        f = kwargs['f'] if 'f' in kwargs else self.forward(x)
        return 1 - f ** 2

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        f = kwargs['f'] if 'f' in kwargs else self.forward(x)
        return 2 * (f ** 3 - f)


class ReLU(Activation):
    def __str__(self) -> str:
        return ReLU.__name__

    def forward(self, x: ndarray) -> ndarray:
        # return np.clip(x, 0, np.inf)
        return np.maximum(x, 0)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, 1, 0)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.zeros_like(x)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.3):
        if isinstance(alpha, str):
            alpha = float(alpha)
        assert isinstance(alpha, float), "Unrecognized alpha type"
        self.alpha = alpha

    def __str__(self) -> str:
        return f"{LeakyReLU.__name__}(alpha={self.alpha})"

    def forward(self, x: ndarray) -> ndarray:
        return np.where(x > 0, x, x * self.alpha)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, 1, self.alpha)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.zeros_like(x)


class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        if isinstance(alpha, str):
            alpha = float(alpha)
        assert isinstance(alpha, float), "Unrecognized alpha type"
        self.alpha = alpha

    def __str__(self) -> str:
        return f"{ELU.__name__}(alpha={self.alpha})"

    def forward(self, x: ndarray) -> ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, 1, self.alpha * np.exp(x))

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, 0, self.alpha * np.exp(x))


class Exponential(Activation):
    def __str__(self) -> str:
        return Exponential.__name__

    def forward(self, x: ndarray) -> ndarray:
        return np.exp(x)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.exp(x)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.exp(x)


class SoftPlus(Activation):
    def __str__(self) -> str:
        return SoftPlus.__name__

    def forward(self, x: ndarray) -> ndarray:
        return np.log(1 + np.exp(x))

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        ex = kwargs['ex'] if 'ex' in kwargs else np.exp(x)
        return ex / (1 + ex)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        ex = kwargs['ex'] if 'ex' in kwargs else np.exp(x)
        return ex / ((1 + ex) ** 2)


class SELU(Activation):
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.elu = ELU(alpha=self.alpha)

    def __str__(self) -> str:
        return SELU.__name__

    def forward(self, x: ndarray) -> ndarray:
        return self.scale * self.elu.forward(x)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(
            x >= 0,
            np.ones_like(x) * self.scale,
            np.exp(x) * self.alpha * self.scale,
            )

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, np.zeros_like(x), np.exp(x) * self.alpha * self.scale)


class HardSigmoid(Activation):
    def __str__(self) -> str:
        return HardSigmoid.__name__

    def forward(self, x: ndarray) -> ndarray:
        return np.clip((0.2 * x) + 0.5, 0.0, 1.0)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where((x >= -2.5) & (x <= 2.5), 0.2, 0)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.zeros_like(x)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes

    def subplot(ax: Axes, x: ndarray, module: Activation):
        ax.plot(inputs, module(x), linestyle='-', label=r"$f(x)$")
        ax.plot(inputs, module.grad(inputs), linestyle='--', label=r"$\dfrac{df}{dx}$")
        ax.plot(inputs, module.grad2(inputs), linestyle=':', label=r"$\dfrac{d^2f}{dx^2}$")
        ax.set_xlim(-5, 5)
        # ax.set_ylim(-0.5, 1.5)
        ax.grid()
        ax.set_title(str(module))
        ax.legend()

    inputs = np.linspace(-5, 5).reshape(-1, 1)

    fig, axs = plt.subplots(2, 5, figsize=[16, 9], dpi=100)
    subplot(axs[0][0], inputs, Sigmoid())
    subplot(axs[0][1], inputs, Tanh())
    subplot(axs[0][2], inputs, ReLU())
    subplot(axs[0][3], inputs, LeakyReLU())
    subplot(axs[0][4], inputs, ELU())
    subplot(axs[1][0], inputs, SoftPlus())
    subplot(axs[1][1], inputs, Affine())
    subplot(axs[1][2], inputs, Exponential())
    subplot(axs[1][3], inputs, SELU())
    subplot(axs[1][4], inputs, HardSigmoid())

    # plt.savefig("./img.png")
    plt.show()
