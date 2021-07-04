from abc import abstractmethod
from numpy import ndarray

from frame.HyperParametersHolder import HyperParametersHolder
from frame.LearningRateScheduler import LearningRateScheduler


class Optimizer(HyperParametersHolder):
    def __init__(self, lr_scheduler: LearningRateScheduler, **kwargs):
        HyperParametersHolder.__init__(
            self,
            lr_scheduler=lr_scheduler,
            step=0,
            cache=[],
            **kwargs
        )

    def __call__(self, parameter_name: str, parameters: ndarray, gradients: ndarray, loss_value: float):
        return self.update(parameter_name, parameters, gradients, loss_value)

    def __str__(self):
        return Optimizer.__name__

    def step(self):
        self['step'] += 1

    def reset_step(self):
        self['step'] = 0

    @abstractmethod
    def update(self, parameter_name: str, parameter: ndarray, gradient: ndarray, loss_value: float):
        """
        根据梯度和损失更新变量

        :param parameter_name: 参数名
        :param parameter: 待更新参数
        :param gradient: 损失函数相对于变量的梯度。
        :param loss_value: 当前批次的损失值
        :return:
        """
        raise NotImplementedError
