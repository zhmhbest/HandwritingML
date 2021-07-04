from typing import Union

import numpy as np
from numpy import ndarray

from frame.Optimizer import Optimizer
from frame.LearningRateScheduler import LearningRateScheduler
from optimizers.learning_rate import \
    ConstantLearningRateScheduler, \
    ExponentialLearningRateScheduler


class SGD(Optimizer):
    def __init__(
            self,
            lr_scheduler: Union[float, LearningRateScheduler] = 0.01,
            momentum: float = 0.0,
            clip_norm: float = None
    ):
        """
        随机梯度下降优化器

        :param lr_scheduler: 学习率变动计划
        :param momentum: 势能 in [0, 1]
        :param clip_norm: 如果不是 None，则在计算更新之前，所有参数梯度都被缩放为具有 `clip_norm` 的最大 l2 范数。
        """
        lr_scheduler = ConstantLearningRateScheduler(lr_scheduler) \
            if isinstance(lr_scheduler, float) else lr_scheduler
        super(SGD, self).__init__(
            lr_scheduler,
            momentum=momentum,
            clip_norm=clip_norm,
            name="SGD"
        )

    def __str__(self):
        return f"{self['name']}" \
               f"(lr_scheduler={self['lr_scheduler']}, momentum={self['momentum']}, clip_norm={self['clip_norm']})"

    def update(self, parameter_name: str, parameter: ndarray, gradient: ndarray, loss_value: float) -> ndarray:
        cache = self['cache']
        momentum = self["momentum"]
        clip_norm = np.inf if self["clip_norm"] is None else self["clip_norm"]
        lr = self['lr_scheduler'](self['step'])

        if parameter_name not in cache:
            cache[parameter_name] = np.zeros_like(gradient)

        # 缩放梯度，避免“梯度消失/梯度爆炸”
        if np.linalg.norm(gradient) > clip_norm:
            gradient = gradient * clip_norm / np.linalg.norm(gradient)

        update = momentum * cache[parameter_name] + lr * gradient
        self['cache'][parameter_name] = update
        return parameter - update
