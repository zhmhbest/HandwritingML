from typing import Union

import numpy as np

from hwml.frame import Optimizer, Parameter, LearningRateScheduler


class RawOptimizer(Optimizer):
    def __init__(self, lr_scheduler: Union[LearningRateScheduler, float]):
        super().__init__(lr_scheduler)

    def update(self, parameter: Parameter):
        parameter.update_parameter(
            self.lr() * parameter.gradient
        )
        parameter.zero_grad()


class SGD(Optimizer):
    def __init__(
            self,
            lr_scheduler: Union[LearningRateScheduler, float],
            momentum: float = 0.0,
            clip_norm: Union[None, float] = None
    ):
        """
        :param lr_scheduler:
        :param momentum: float in range [0, 1]
        :param clip_norm:
        """
        super().__init__(lr_scheduler)
        self.momentum = momentum
        self.clip_norm = clip_norm

    def update(self, parameter: Parameter):
        lr = self.lr()
        gradient = parameter.gradient
        cache = parameter.get_cache("deviation", np.zeros_like(parameter.gradient))

        # scale gradient to avoid explosion
        t = np.inf if self.clip_norm is None else self.clip_norm
        if np.linalg.norm(gradient) > t:
            gradient = gradient * t / np.linalg.norm(gradient)

        deviation = self.momentum * cache + lr * gradient
        parameter.set_cache("deviation", deviation)

        parameter.update_parameter(deviation)
        parameter.zero_grad()


class Adam(Optimizer):
    def __init__(
            self,
            lr_scheduler: Union[LearningRateScheduler, float],
            decay1: float = 0.9,
            decay2: float = 0.999,
            clip_norm: Union[None, float] = None,
            eps: float = 1e-7
    ):
        """
        :param lr_scheduler:
        :param decay1: decay for avg
        :param decay2: decay for var
        :param clip_norm:
        :param eps:
        """
        super().__init__(lr_scheduler)
        self.decay1 = decay1
        self.decay2 = decay2
        self.clip_norm = clip_norm
        self.eps = eps

    def update(self, parameter: Parameter):
        lr = self.lr()
        gradient = parameter.gradient
        cache_t = parameter.get_cache("t", 0)
        cache_avg = parameter.get_cache("avg", np.zeros_like(parameter.gradient))
        cache_var = parameter.get_cache("var", np.zeros_like(parameter.gradient))

        # scale gradient to avoid explosion
        t = np.inf if self.clip_norm is None else self.clip_norm
        if np.linalg.norm(gradient) > t:
            gradient = gradient * t / np.linalg.norm(gradient)

        # update cache
        cache_t += 1
        cache_avg = self.decay1 * cache_avg + (1 - self.decay1) * gradient
        cache_var = self.decay2 * cache_var + (1 - self.decay2) * gradient ** 2
        parameter.set_cache("t", cache_t)
        parameter.set_cache("avg", cache_avg)
        parameter.set_cache("var", cache_var)

        # calc unbiased moment estimates and Adam update
        hat_avg = cache_avg / (1 - self.decay1 ** t)
        hat_var = cache_var / (1 - self.decay2 ** t)
        deviation = lr * hat_avg / (np.sqrt(hat_var) + self.eps)

        parameter.update_parameter(deviation)
        parameter.zero_grad()
