from abc import abstractmethod
from typing import List, Dict, Union, Iterator, Tuple

from numpy import ndarray

from .optimizer import Parameter, Optimizer


class Layer:
    def __init__(self):
        # 是否保留衍生变量（请在forward时判断）
        self.retain_derived = True
        # 衍生变量列表
        self.derivations = {'X': []}

    def define_derivation(self, key):
        """定义衍生变量"""
        self.derivations[key] = []

    def clear_derivations(self):
        """清空衍生变量记录"""
        for key in self.derivations:
            self.derivations[key].clear()

    def iterable_pl_pz_x(self, pl_pz_list: List[ndarray]) -> Iterator[Tuple[ndarray, ndarray]]:
        """
        在定义了衍生变量X的情况下，返回可遍历的列表(pl_pz, x)
        """
        if 'X' not in self.derivations.keys():
            raise ValueError("Undefined 'X' in derivations.")
        x_list = self.derivations['X']
        if len(x_list) != len(pl_pz_list):
            raise BufferError("The derived quantity does not match the partial derivative quantity.")
        return zip(pl_pz_list, x_list)

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    def __call__(self, x: ndarray) -> ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        """前向传播"""
        raise NotImplementedError()

    @abstractmethod
    def calc_gradients(self, pl_pz: ndarray, x: ndarray, index: int) -> ndarray:
        """
        计算本层所有参数的梯度（如果有）
        计算损失对本层输入的梯度
        :param pl_pz:
        :param x:
        :param index: 标记衍生变量顺序
        :return: 损失对本层输入的梯度
        """
        raise NotImplementedError()

    def backward(self, pl_pz_list: List[ndarray]) -> List[ndarray]:
        """
        反向传播
            计算所有参数的梯度
            传播完成后清除衍生变量
        :param pl_pz_list: 损失对本层输出的梯度
        :return: 损失对本层输入的梯度
        """
        pl_px_list = []
        index = 0
        for pl_pz, x in self.iterable_pl_pz_x(pl_pz_list):
            pl_px_list.append(self.calc_gradients(pl_pz, x, index))
            index += 1
        self.clear_derivations()
        return pl_px_list


class FunctionLayer(Layer):
    """
    无参数的Layer
    """
    def calc_gradients(self, pl_pz: ndarray, x: ndarray, index: int) -> ndarray:
        """（反向传播）函数对输入的梯度"""
        return pl_pz * self.grad(x)

    @abstractmethod
    def grad(self, x: ndarray, **kwargs) -> ndarray:
        """（本质属性）函数对输入的梯度"""
        raise NotImplementedError()

    def __call__(self, x: ndarray):
        """弃用forward，改用fn"""
        return self.fn(x)

    @abstractmethod
    def fn(self, x: ndarray) -> ndarray:
        """弃用forward，改用fn"""
        raise NotImplementedError()

    def forward(self, x: ndarray) -> ndarray:
        """弃用forward，改用fn"""
        return self.fn(x)


class ParameterLayer(Layer):
    """
    有可训练参数的Layer
    """

    def __init__(self):
        super().__init__()
        # 优化器
        self.optimizer: Union[None, Optimizer] = None
        # 参数列表
        self.parameters: Dict[str, Parameter] = {}

    def define_parameter(self, key, **kwargs):
        """定义参数"""
        self.parameters[key] = Parameter(**kwargs)

    def get_parameter(self, key):
        """获取参数"""
        return self.parameters[key]

    def update_parameters(self):
        """更新所有参数"""
        assert self.optimizer is not None, "Undefined optimizer."
        self.optimizer.update_dict_parameters(self.parameters)

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        raise NotImplementedError()

    @abstractmethod
    def calc_gradients(self, pl_pz: ndarray, x: ndarray, index: int) -> ndarray:
        raise NotImplementedError()

    def backward(self, pl_pz_list: List[ndarray]) -> List[ndarray]:
        """
        反向传播
            计算所有参数的梯度
            传播完成后清除衍生变量
            计算梯度后更新参数
        """
        pl_px_list = Layer.backward(self, pl_pz_list)
        self.update_parameters()
        return pl_px_list
