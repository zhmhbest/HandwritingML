from abc import abstractmethod
from numpy import ndarray
import numpy as np
from optimizers import OptimizerBase


class Freezable:
    def __init__(self):
        self.trainable = True

    def freeze(self):
        """冻结"""
        self.trainable = False

    def unfreeze(self):
        """解冻"""
        self.trainable = True


class WeightHolder:
    def __init__(self):
        # 超参数
        self.hyper_parameters = {}
        # 权重
        self.parameters = {}
        # 梯度
        self.gradients = {}
        # 前向传递期间计算的衍生变量（输入、输出、输入维度、输出维度等信息）
        self.derived_variables = {
            'X': [],
        }
        # 是否初始化
        self.is_initialized = False

    def _init_params(self, **kwargs):
        """初始化参数"""
        keys = kwargs.keys()
        if 'hyper_parameters' in keys:
            self.hyper_parameters.update(kwargs['hyper_parameters'])
        if 'parameters' in keys:
            self.parameters.update(kwargs['parameters'])
        if 'gradients' in keys:
            self.gradients.update(kwargs['gradients'])
        if 'derived_variables' in keys:
            self.derived_variables.update(kwargs['derived_variables'])
        self.is_initialized = True

    def __setitem__(self, hyper_parameter_name, value):
        """[] 设置超参数"""
        self.hyper_parameters[hyper_parameter_name] = value

    def __getitem__(self, hyper_parameter_name: str):
        """[] 返回超参数"""
        return self.hyper_parameters[hyper_parameter_name]

    def _zero_gradients(self):
        """重置 梯度及衍生变量"""
        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

    def _update_gradients(self, optimizer: OptimizerBase, loss):
        """使用优化器和累计梯度更新权重"""
        optimizer.step()
        for gk, gv in self.gradients.items():
            if gk in self.parameters:
                self.parameters[gk] = optimizer(self.parameters[gk], gv, gk, loss)
        self._zero_gradients()

    def _summary(self, **kwargs) -> dict:
        result = {
            'parameters': self.parameters,
            'hyper_parameters': self.hyper_parameters
        }
        result.update(kwargs)
        return result


class LayerBase(Freezable, WeightHolder):
    def __init__(self, layer_name: str = "LayerBase"):
        Freezable.__init__(self)
        WeightHolder.__init__(self)
        self.layer_name = layer_name

    def __str__(self):
        return self.layer_name

    def zero_gradients(self):
        assert self.trainable, "Layer is frozen"
        WeightHolder._zero_gradients(self)

    def update_gradients(self, optimizer: OptimizerBase, loss):
        assert self.trainable, "Layer is frozen"
        WeightHolder._update_gradients(self, optimizer, loss)

    def summary(self):
        return WeightHolder._summary(self, name=self.layer_name)

    @abstractmethod
    def forward(self, x: ndarray, **kwargs):
        """前向传播"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, out: ndarray, **kwargs):
        """反向传播"""
        raise NotImplementedError
