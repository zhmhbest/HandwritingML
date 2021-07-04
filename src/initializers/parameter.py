import re
from typing import Tuple, Union

import numpy as np


def get_params_io(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    根据参数Shape计算输入输出尺寸
    :param shape: 参数Shape
    :returns: (输入尺寸, 输出尺寸)
    """
    shape_dim = len(shape)
    if 2 == shape_dim:
        # (I, O) = (int, int)
        in_dim, out_dim = shape
        return in_dim, out_dim
    elif shape_dim in [3, 4]:
        # (?, I, O) = (int, int, int)
        # (?, ?, I, O) = (int, int, int, int)
        in_dim, out_dim = shape[-2:]
        kernel_size = np.prod(shape[:-2])
        return in_dim * kernel_size, out_dim * kernel_size
    else:
        raise ValueError(f"Unrecognized weight dimension: {shape}")


def he_uniform(shape: Tuple[int, ...]):
    _i_, _o_ = get_params_io(shape)
    _b_ = np.sqrt(6 / _i_)
    return np.random.uniform(-_b_, _b_, size=shape)


def he_normal(shape: Tuple[int, ...]):
    _i_, _o_ = get_params_io(shape)
    _std_ = np.sqrt(2 / _i_)
    return truncated_normal(shape, 0, _std_)


def glorot_uniform(shape: Tuple[int, ...], gain: float = 1.0):
    _i_, _o_ = get_params_io(shape)
    _b_ = gain * np.sqrt(6 / (_i_ + _o_))
    return np.random.uniform(-_b_, _b_, size=shape)


def glorot_normal(shape: Tuple[int, ...], gain: float = 1.0):
    _i_, _o_ = get_params_io(shape)
    _std_ = gain * np.sqrt(2 / (_i_ + _o_))
    return truncated_normal(shape, 0, _std_)


def truncated_normal(shape: Tuple[int, ...], mean: float = 0, std: float = 1):
    """生成2σ内的高斯分布"""
    # 生成高斯分布
    get_samples = (lambda current_shape: np.random.normal(loc=mean, scale=std, size=current_shape))
    # 分布超出2σ的部分
    get_rejects = (lambda current_samples: np.greater_equal(np.abs(current_samples) - mean, std * 2))
    #
    dump = get_samples(shape)
    rejects = get_rejects(dump)
    rejects_sum = rejects.sum()
    while rejects_sum > 0:
        samples = get_samples(rejects_sum)
        dump[rejects] = samples
        rejects = get_rejects(dump)
        rejects_sum = rejects.sum()
    return dump


_initializers = {
    # Default
    "": truncated_normal,
    "he_normal": he_normal,
    "he_uniform": he_uniform,
    "glorot_normal": glorot_normal,
    "glorot_uniform": glorot_uniform,
    "std_normal": (lambda shape: np.random.randn(*shape)),
    "trunc_normal": truncated_normal,
}


class ParametersInitializer:
    def __init__(self, mode: str = "", activation_name: str = ""):
        """
        参数初始化工厂。

        Parameters
        ----------
        mode : str (default: 'glorot_uniform')
            权重初始化策略
        activation_name : str
            层激活函数的字符串表示名称
        """
        if mode not in _initializers.keys():
            raise ValueError(f"Unrecognized initialization mode: {mode}")
        self.mode = mode
        self.activation_name = activation_name.lower().replace(" ", "_")
        self._fn = _initializers[mode]

    def __call__(self, weight_shape):
        if self.mode.startswith("glorot"):
            gain = self.calc_glorot_gain()
            w = self._fn(weight_shape, gain)
        else:
            w = self._fn(weight_shape)
        return w

    def calc_glorot_gain(self) -> Union[float, np.ndarray]:
        """
        https://pytorch.org/docs/stable/nn.html?#torch.nn.init.calculate_gain
        """
        if "tanh" == self.activation_name:
            return 5.0 / 3.0
        elif "relu" == self.activation_name:
            return np.sqrt(2)
        elif "leaky_relu" == self.activation_name:
            alpha = float(re.match(r"leaky_relu\(alpha=(.+)\)", self.activation_name).groups()[0])
            return np.sqrt(2 / 1 + alpha ** 2)
        else:
            return 1.0


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from numpy import ndarray

    def subplot(ax: Axes, initializer_name: str):
        global RANGE_LENGTH
        initializer = _initializers[initializer_name]
        # Scatter
        for i in range(50):
            ax.scatter(range(RANGE_LENGTH), initializer((1, RANGE_LENGTH)), s=0.1, c="pink")
        # 辅助线
        ax.plot([0 for i in range(RANGE_LENGTH)], linestyle=":")
        if initializer_name.startswith("glorot"):
            ax.set_ylim(-0.1, 0.1)
        elif initializer_name.startswith("trunc"):
            ax.plot([2 for i in range(RANGE_LENGTH)], linestyle=":")
            ax.plot([-2 for i in range(RANGE_LENGTH)], linestyle=":")
            ax.set_ylim(-3, 3)
        else:
            ax.set_ylim(-4, 4)
        ax.set_title(initializer_name)

    RANGE_LENGTH = 1000
    fig, axs = plt.subplots(2, 3, figsize=[16, 9], dpi=100)
    subplot(axs[0][0], "he_normal")
    subplot(axs[0][1], "he_uniform")
    subplot(axs[0][2], "std_normal")
    subplot(axs[1][0], "trunc_normal")
    subplot(axs[1][1], "glorot_normal")
    subplot(axs[1][2], "glorot_uniform")
    plt.show()
