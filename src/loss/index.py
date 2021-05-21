import numpy as np


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    :param y_true: 已经过OneHot编码
    :param y_pred: 已转化为概率分布（即已经过Softmax处理）
    """
    return -np.sum(y_true * np.log(y_pred), axis=1)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    :param y_true: 取值仅为 0 | 1
    :param y_pred: 已转化为概率分布（即已经过Sigmoid处理）
    """
    if 2 != len(y_true.shape) or 2 != len(y_pred.shape):
        raise Exception("Unmatched dimensions.")
    add_hidden_dimension = (lambda x: np.hstack([x, 1 - x]))
    if 1 == y_true.shape[1]:
        y_true = add_hidden_dimension(y_true)
    if 1 == y_pred.shape[1]:
        y_pred = add_hidden_dimension(y_pred)
    return categorical_cross_entropy(y_true, y_pred)


if __name__ == '__main__':
    t_true = np.random.randint(0, 2, size=(10, 1)).astype(np.float)
    t_pred = np.random.uniform(0, 1, size=(10, 1)).astype(np.float)
    t_loss = binary_cross_entropy(t_true, t_pred).reshape(-1, 1)
    print(np.hstack([
        t_true, t_pred, t_loss
    ]))
    print(t_loss.mean())
