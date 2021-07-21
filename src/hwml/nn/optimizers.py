from numpy import ndarray

from hwml.frame import Optimizer


class RawOptimizer(Optimizer):
    def __init__(self, lr_scheduler):
        super().__init__(lr_scheduler)

    def update(self, parameter: ndarray, gradient: ndarray, **kwargs):
        return parameter - self.lr_scheduler(self.num_step) * gradient
