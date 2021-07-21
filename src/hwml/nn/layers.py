import numpy as np
from numpy import ndarray

from hwml.frame import ParameterLayer


class Linear(ParameterLayer):
    def __init__(self, input_dim: int, output_dim: int, retain_derived: bool = True):
        super().__init__()
        # 参数
        self.io_shape = (input_dim, output_dim)
        self.define_parameter('w', shape=self.io_shape, initializer='he_uniform')
        self.define_parameter('b', shape=(1, output_dim), initializer='zeros')

        # 衍生值
        self.retain_derived = retain_derived
        self.define_derivation('X')

    def __str__(self) -> str:
        return Linear.__name__

    def forward(self, x: ndarray) -> ndarray:
        if self.retain_derived:
            self.derivations['X'].append(x)
        return x @ self.get_parameter('w')() + self.get_parameter('b')()

    def calc_gradient(self, pl_pz: ndarray, x: ndarray) -> ndarray:
        pl_px = pl_pz @ self.get_parameter('w')().T
        self.get_parameter('w').accumulate_grad(x.T @ pl_pz)
        self.get_parameter('b').accumulate_grad(np.sum(pl_pz, axis=0, keepdims=True))
        return pl_px
