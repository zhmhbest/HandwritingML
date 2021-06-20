from typing import Tuple
import numpy as np
from functools import partial


def get_weight_io_dimension(weight_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    根据参数shape计算输入输出维度

    :param weight_shape: tuple
    :returns: in_dim, out_dim
    """
    weight_dim_size = len(weight_shape)
    if 2 == weight_dim_size:
        # (int, int)
        in_dim, out_dim = weight_shape
        return in_dim, out_dim
    elif weight_dim_size in [3, 4]:
        in_dim, out_dim = weight_shape[-2:]
        kernel_size = np.prod(weight_shape[:-2])
        return in_dim * kernel_size, out_dim * kernel_size
    else:
        raise ValueError(f"Unrecognized weight dimension: {weight_shape}")


def he_uniform(weight_shape: Tuple[int, ...]):
    in_dim, out_dim = get_weight_io_dimension(weight_shape)
    b = np.sqrt(6 / in_dim)
    return np.random.uniform(-b, b, size=weight_shape)


def he_normal(weight_shape: Tuple[int, ...]):
    in_dim, out_dim = get_weight_io_dimension(weight_shape)
    std = np.sqrt(2 / in_dim)
    return truncated_normal(0, std, weight_shape)


def glorot_uniform(weight_shape: Tuple[int, ...], gain: float = 1.0):
    in_dim, out_dim = get_weight_io_dimension(weight_shape)
    b = gain * np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(-b, b, size=weight_shape)


def glorot_normal(weight_shape: Tuple[int, ...], gain: float = 1.0):
    in_dim, out_dim = get_weight_io_dimension(weight_shape)
    std = gain * np.sqrt(2 / (in_dim + out_dim))
    return truncated_normal(0, std, weight_shape)


def truncated_normal(mean: float, std: float, out_shape: Tuple[int, ...]):
    samples = np.random.normal(loc=mean, scale=std, size=out_shape)
    reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    while any(reject.flatten()):
        resamples = np.random.normal(loc=mean, scale=std, size=reject.sum())
        samples[reject] = resamples
        reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    return samples


weight_initializers = {
    "he_normal": he_normal,
    "he_uniform": he_uniform,
    "glorot_normal": glorot_normal,
    "glorot_uniform": glorot_uniform,
    "std_normal": np.random.randn,
    "trunc_normal": partial(truncated_normal, mean=0, std=1),
}


class WeightInitializer:
    def __init__(self, activation: str, mode: str = "glorot_uniform"):
        """
        权重初始化工厂。

        Parameters
        ----------
        activation : str
            层激活函数的字符串表示名称
        mode : str (default: 'glorot_uniform')
            权重初始化策略
        """
        if mode not in weight_initializers.keys():
            raise ValueError(f"Unrecognized initialization mode: {mode}")
        self.mode = mode
        self.activation = activation.lower().replace(' ', '_')
        self._fn = weight_initializers[mode]

    def __call__(self, weight_shape):
        if self.mode.startswith("glorot"):
            gain = self._calc_glorot_gain()
            w = self._fn(weight_shape, gain)
        elif "std_normal" == self.mode:
            w = self._fn(*weight_shape)
        else:
            w = self._fn(weight_shape)
        return w

    def _calc_glorot_gain(self):
        """
        https://pytorch.org/docs/stable/nn.html?#torch.nn.init.calculate_gain
        """
        import re
        gain = 1.0
        if "tanh" == self.activation:
            gain = 5.0 / 3.0
        elif "relu" == self.activation:
            gain = np.sqrt(2)
        elif "leaky_relu" == self.activation:
            alpha = re.match(r"leaky_relu\(alpha=(.+)\)", self.activation).groups()[0]
            alpha = float(alpha)
            gain = np.sqrt(2 / 1 + alpha ** 2)
        return gain
