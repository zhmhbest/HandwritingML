from abc import abstractmethod

from frame.HyperParametersHolder import HyperParametersHolder


class LearningRateScheduler(HyperParametersHolder):
    def __init__(self, **kwargs):
        HyperParametersHolder.__init__(self, **kwargs)

    def __call__(self, step: int) -> float:
        return self.learning_rate(step)

    @abstractmethod
    def learning_rate(self, step: int) -> float:
        raise NotImplementedError
