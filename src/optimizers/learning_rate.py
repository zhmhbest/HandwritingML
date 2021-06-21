from abc import abstractmethod

import numpy as np
from numpy import ndarray


def gaussian_cdf(x: ndarray, mean: ndarray, var: ndarray):
    """
    高斯分布概率密度函数
    `mean` & `var` <= `x`.
    """
    from math import erf
    eps = np.finfo(float).eps  # 很小的非负数，用来防止除数为0
    x_scaled = (x - mean) / np.sqrt(var + eps)
    return (1 + erf(x_scaled / np.sqrt(2))) / 2


class LearningRateScheduler:
    def __init__(self, **kwargs):
        self.hyper_parameters = kwargs

    def __call__(self, step: int = None):
        return self.learning_rate(step)

    def __getitem__(self, key):
        return self.hyper_parameters[key]

    def __setitem__(self, key, value):
        self.hyper_parameters[key] = value

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    @abstractmethod
    def learning_rate(self, step: int):
        raise NotImplementedError


class ConstantLearningRateScheduler(LearningRateScheduler):
    def __init__(self, lr=0.01):
        super().__init__(
            lr=lr,
            name="ConstantLearningRate"
        )

    def __str__(self):
        return f"{self['name']}(lr={self['lr']})"

    def learning_rate(self, step):
        return self['lr']


class ExponentialLearningRateScheduler(LearningRateScheduler):
    def __init__(self, initial_lr: float = 0.01, stage_length: int = 500, staircase: bool = False, decay: float = 0.1):
        """
        指数衰减法

        :param initial_lr: 初始学习率
        :param stage_length: 每个阶段的长度（单位为步）
        :param staircase: 是否仅在阶段转换时调整学习速率
        :param decay: 每一阶段学习率衰减量
        """
        super().__init__(
            initial_lr=initial_lr,
            stage_length=stage_length,
            staircase=staircase,
            decay=decay,
            name="ExponentialLearningRate"
        )

    def __str__(self):
        return f"{self['name']}(initial_lr={self['initial_lr']}, stage_length={self['stage_length']}, staircase={self['staircase']}, decay={self['decay']})"

    def learning_rate(self, step):
        cur_stage = step / self['stage_length']
        if self['staircase']:
            cur_stage = np.floor(cur_stage)
        return self['initial_lr'] * self['decay'] ** cur_stage

