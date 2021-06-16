import re
import numpy as np
from utils import calc_fan
from functools import partial


class InitializerBase:
    pass


#######################################################################
#                        Weight Initialization                        #
#######################################################################


def he_uniform(weight_shape):
    fan_in, fan_out = calc_fan(weight_shape)
    b = np.sqrt(6 / fan_in)
    return np.random.uniform(-b, b, size=weight_shape)


def he_normal(weight_shape):
    fan_in, fan_out = calc_fan(weight_shape)
    std = np.sqrt(2 / fan_in)
    return truncated_normal(0, std, weight_shape)


def glorot_uniform(weight_shape, gain=1.0):
    fan_in, fan_out = calc_fan(weight_shape)
    b = gain * np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-b, b, size=weight_shape)


def glorot_normal(weight_shape, gain=1.0):
    fan_in, fan_out = calc_fan(weight_shape)
    std = gain * np.sqrt(2 / (fan_in + fan_out))
    return truncated_normal(0, std, weight_shape)


def truncated_normal(mean, std, out_shape):
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


class WeightInitializer(InitializerBase):
    def __init__(self, act_fn_str, mode="glorot_uniform"):
        """
        重量初始化器的工厂。

        Parameters
        ----------
        act_fn_str : str
            层激活函数的字符串表示名称
        mode : str (default: 'glorot_uniform')
            权重初始化策略
        """
        if mode not in weight_initializers.keys():
            raise ValueError(f"Unrecognized initialization mode: {mode}")
        self.mode = mode
        self.act_fn = act_fn_str
        self._fn = weight_initializers[mode]

    def __call__(self, weight_shape):
        if "glorot" in self.mode:
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
        gain = 1.0
        act_str = self.act_fn.lower()
        if act_str == "tanh":
            gain = 5.0 / 3.0
        elif act_str == "relu":
            gain = np.sqrt(2)
        elif "leaky relu" in act_str:
            r = r"leaky relu\(alpha=(.*)\)"
            alpha = re.match(r, act_str).groups()[0]
            gain = np.sqrt(2 / 1 + float(alpha) ** 2)
        return gain
