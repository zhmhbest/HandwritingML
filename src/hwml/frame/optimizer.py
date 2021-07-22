from typing import Tuple, Union, Dict

import numpy as np
from numpy import ndarray

from .scheduler import LearningRateScheduler
from ..nn.schedulers import ConstantLRS


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
    elif 3 == shape_dim or 4 == shape_dim:
        # (?, I, O) = (int, int, int)
        # (?, ?, I, O) = (int, int, int, int)
        in_dim, out_dim = shape[-2:]
        kernel_size = np.prod(shape[:-2])
        return in_dim * kernel_size, out_dim * kernel_size
    else:
        raise ValueError(f"Unrecognized dimension: {shape}")


def he_uniform(shape: Tuple[int, ...]):
    i_dim, o_dim = get_params_io(shape)
    b = np.sqrt(6 / i_dim)
    return np.random.uniform(-b, b, size=shape)


def he_normal(shape: Tuple[int, ...]):
    i_dim, o_dim = get_params_io(shape)
    std = np.sqrt(2 / i_dim)
    return truncated_normal(shape, 0, std)


def glorot_uniform(shape: Tuple[int, ...], gain: float = 1.0):
    i_dim, o_dim = get_params_io(shape)
    b = gain * np.sqrt(6 / (i_dim + o_dim))
    return np.random.uniform(-b, b, size=shape)


def glorot_normal(shape: Tuple[int, ...], gain: float = 1.0):
    i_dim, o_dim = get_params_io(shape)
    std = gain * np.sqrt(2 / (i_dim + o_dim))
    return truncated_normal(shape, 0, std)


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
    "he_normal": he_normal,
    "he_uniform": he_uniform,
    "glorot_normal": glorot_normal,
    "glorot_uniform": glorot_uniform,
    "std_normal": (lambda shape: np.random.randn(*shape)),
    "trunc_normal": truncated_normal,
    "zeros": np.zeros,
}


class Parameter:
    def __init__(self, shape: Tuple[int, ...], initializer: str = "", **kwargs):
        """
        可训练参数

        :param shape: 参数形状
        :param initializer: 参数初始化方法
        :param kwargs: 参数初始化方法附加参数
        """
        initializer = initializer.strip().lower()
        if initializer not in _initializers.keys():
            raise ValueError(
                f"Unrecognized initializer '{initializer}', "
                f"it has to be one of [{', '.join(_initializers.keys())}]."
            )
        # 形状
        self.shape = shape
        # 参数
        self.parameter: ndarray = _initializers[initializer](shape, **kwargs)
        # 梯度
        self.gradient: ndarray = np.zeros_like(self.parameter)
        # 缓存
        self.cache: Dict[str, Union[ndarray, int, float]] = {}

    def __call__(self) -> ndarray:
        """前向传播时调用参数"""
        return self.parameter

    def set_cache(self, key, value: Union[ndarray, int, float]):
        """设置缓存"""
        self.cache[key] = value

    def get_cache(self, key: str, value: Union[ndarray, int, float] = None):
        """获取缓存"""
        if key in self.cache.keys():
            return self.cache[key]
        else:
            self.cache[key] = value
            return value

    def zero_grad(self):
        """重置梯度"""
        self.gradient = np.zeros_like(self.parameter)

    def accumulate_grad(self, grad: ndarray):
        """积累梯度"""
        self.gradient += grad

    def update_parameter(self, deviation: ndarray):
        """更新参数值"""
        self.parameter -= deviation


class Optimizer:
    def __init__(self, lr_scheduler: Union[LearningRateScheduler, float]):
        # 学习率曲线
        self.lr_scheduler = ConstantLRS(lr_scheduler) if isinstance(lr_scheduler, float) else lr_scheduler
        # 迭代次数
        self.num_step: int = 0

    def step(self):
        self.num_step += 1

    def reset(self):
        self.num_step = 0

    def lr(self) -> float:
        """学习率"""
        return self.lr_scheduler(self.num_step)

    def update_dict_parameters(self, parameters: Dict[str, Parameter]):
        """更新参数表中的参数"""
        for key in parameters:
            self.update(parameters[key])

    def update(self, parameter: Parameter):
        """更新参数"""
        raise NotImplementedError()
