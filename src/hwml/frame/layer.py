from abc import abstractmethod
from typing import List, Dict, Union

from numpy import ndarray

from .optimizer import Parameter, Optimizer


class Layer:
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    def __call__(self, x: ndarray) -> ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        """前向传播"""
        raise NotImplementedError()


class FunctionLayer(Layer):
    """
    无参数的Layer
    """
    @abstractmethod
    def grad(self, x: ndarray, **kwargs) -> ndarray:
        """计算梯度"""
        raise NotImplementedError()


class ActivationLayer(FunctionLayer):
    """
    激活函数
    """
    @abstractmethod
    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        """计算梯度的梯度"""
        raise NotImplementedError()


class ParameterLayer(Layer):
    """
    拥有可训练参数的Layer
    """

    def __init__(self):
        # 优化器
        self.optimizer: Union[None, Optimizer] = None
        # 参数列表
        self.parameters: Dict[str, Parameter] = {}
        # 衍生变量列表
        self.derivations = {}

    def define_derivation(self, key):
        """定义衍生变量"""
        self.derivations[key] = []

    def define_parameter(self, key, **kwargs):
        """定义参数"""
        self.parameters[key] = Parameter(**kwargs)

    def get_parameter(self, key):
        """获取参数"""
        return self.parameters[key]

    def clear_derivations(self):
        """清空衍生变量记录"""
        for key in self.derivations:
            self.derivations[key].clear()

    @abstractmethod
    def calc_gradient(self, pl_pz: ndarray, x: ndarray) -> ndarray:
        raise NotImplementedError()

    def backward(self, pl_pz_list: List[ndarray]) -> List[ndarray]:
        """
        反向传播
            计算所有参数的梯度
            传播完成后清除衍生变量
            计算梯度后更新参数
        :param pl_pz_list: 损失对本层输出的梯度
        :return: 损失对本层输入的梯度
        """
        x_list = self.derivations['X']
        if len(x_list) != len(pl_pz_list):
            raise BufferError("The derived quantity does not match the partial derivative quantity.")

        pl_px_list = []
        for pl_pz, x in zip(pl_pz_list, x_list):
            pl_px_list.append(self.calc_gradient(pl_pz, x))

        self.clear_derivations()
        assert self.optimizer is not None, "Undefined optimizer."
        self.optimizer.update_dict_parameters(self.parameters)
        return pl_px_list
