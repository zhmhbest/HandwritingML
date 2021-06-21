from typing import Union
from optimizers.learning_rate import *
from numpy import ndarray


class OptimizerBase:
    def __init__(self, lr: Union[float, LearningRateScheduler], **kwargs):
        self.hyper_parameters = {
            'step': 0,  # 迭代次数
            'lr_scheduler': ConstantLearningRateScheduler(lr) if isinstance(lr, float) else lr,
            'cache': []
        }
        self.hyper_parameters.update(kwargs)

    def __setitem__(self, key, value):
        self.hyper_parameters[key] = value

    def __getitem__(self, key):
        return self.hyper_parameters[key]

    def __call__(self, parameter: ndarray, gradient: ndarray, parameter_name: str, loss_value: float):
        return self.update(parameter, gradient, parameter_name, loss_value)

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def step(self):
        self['step'] += 1

    def reset_step(self):
        self['step'] = 0

    @abstractmethod
    def update(self, parameter: ndarray, gradient: ndarray, parameter_name: str, loss_value: float):
        """
        根据梯度和损失更新变量

        :param parameter: 待更新的变量
        :param gradient: 损失函数相对于变量的梯度。
        :param parameter_name: 参数名称
        :param loss_value: 当前批次的损失值
        :return:
        """
        raise NotImplementedError


class SGD(OptimizerBase):
    def __init__(self, lr: Union[float, LearningRateScheduler] = 0.01, momentum: float = 0.0, clip_norm: float = None):
        """
        随机梯度下降优化器
        :param lr:
        :param momentum: float in range [0, 1]
        :param clip_norm: float, 如果不是 None，则在计算更新之前，所有参数梯度都被缩放为具有 `clip_norm` 的最大 l2 范数。
        """
        super().__init__(lr, momentum=momentum, clip_norm=clip_norm, name="SGD")

    def __str__(self):
        return f"{self['name']}" \
               f"(lr_scheduler={self['lr_scheduler']}, momentum={self['momentum']}, clip_norm={self['clip_norm']})"

    def update(self, parameter: ndarray, gradient: ndarray, parameter_name: str, loss_value: float):
        cache = self['cache']
        lr_scheduler = self['lr_scheduler']
        lr, momentum = lr_scheduler(self['step']), self["momentum"]
        clip_norm = np.inf if self["clip_norm"] is None else self["clip_norm"]

        if parameter_name not in cache:
            cache[parameter_name] = np.zeros_like(gradient)

        # scale gradient to avoid explosion
        if np.linalg.norm(gradient) > clip_norm:
            gradient = gradient * clip_norm / np.linalg.norm(gradient)

        update = momentum * cache[parameter_name] + lr * gradient
        self['cache'][parameter_name] = update
        return parameter - update
