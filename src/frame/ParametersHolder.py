import numpy as np
from frame.HyperParametersHolder import HyperParametersHolder
from frame.Optimizer import Optimizer


class ParametersHolder(HyperParametersHolder):
    def __init__(self, **kwargs):
        # 超参数
        HyperParametersHolder.__init__(self, **kwargs)
        # 权重
        self.parameters = {}
        # 梯度
        self.gradients = {}
        # 前向传递期间计算的衍生变量（输入、输出、输入维度、输出维度等信息）
        self.derived_variables = {}
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
        else:
            # 与参数同步
            for k, v in self.parameters.items():
                self.gradients[k] = np.zeros_like(v)

        if 'derived_variables' in keys:
            self.derived_variables.update(kwargs['derived_variables'])
        self.is_initialized = True

    def _zero_gradients(self):
        """重置 梯度及衍生变量"""
        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

    def _update_gradients(self, optimizer: Optimizer, loss_value: float):
        """使用优化器和累计梯度更新权重"""
        optimizer.step()
        for gk, gv in self.gradients.items():
            if gk in self.parameters:
                # 参数名, 参数, 梯度, 损失值
                self.parameters[gk] = optimizer(gk, self.parameters[gk], gv, loss_value)
        self._zero_gradients()

    def get_parameters(self) -> dict:
        return self.parameters

    def get_gradients(self) -> dict:
        return self.gradients

    def get_summary(self, **kwargs) -> dict:
        """获得 超参数、参数"""
        result = {
            'hyper_parameters': self.hyper_parameters,
            'parameters': self.parameters
        }
        result.update(kwargs)
        return result
