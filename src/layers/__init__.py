
# def softmax(x: np.ndarray) -> np.ndarray:
#     es = np.exp(x)
#     return es / np.sum(es, axis=1, keepdims=True)
#
#
# def stable_softmax(x: np.ndarray) -> np.ndarray:
#     es = np.exp(x - np.max(x))
#     return es / np.sum(es, axis=1, keepdims=True)
#
#
# def distill_softmax(x: np.ndarray, t: int = 2) -> np.ndarray:
#     es = np.exp(x / t)
#     return es / np.sum(es, axis=1, keepdims=True)

# from abc import abstractmethod
# from numpy import ndarray
# import numpy as np
#
# from base import BaseModule
