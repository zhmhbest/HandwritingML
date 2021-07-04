from abc import abstractmethod
from typing import Union

from numpy import ndarray


class Loss:
    def __str__(self):
        return Loss.__name__

    def __call__(self, y_true: ndarray, y_pred: ndarray) -> Union[ndarray, float]:
        return self.loss(y_true, y_pred)

    @abstractmethod
    def loss(self, y_true: ndarray, y_pred: ndarray) -> Union[ndarray, float]:
        raise NotImplementedError

    @staticmethod
    def grad(self, y_true: ndarray, y_pred: ndarray, **kwargs) -> ndarray:
        raise NotImplementedError
