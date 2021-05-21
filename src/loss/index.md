# Loss

[TOC]

## 回归问题

### MAE

$$\mathrm{MAE}(y, f(x)) = \dfrac{\sum\nolimits_{i=1}^{n}|y_i-f(x_i)|}{n}$$

### MSE

$$\mathrm{MSE}(y, f(x)) = \dfrac{\sum\nolimits_{i=1}^{n}(y_i-f(x_i))^2}{n}$$

### RMSE

$$\mathrm{RMSE}(y, f(x)) = \sqrt{ \dfrac{\sum\nolimits_{i=1}^{n}(y_i-f(x_i))^2}{n}}$$

### Huber

>建议$δ=1.35$以达到$95\%$的有效性。

$$L_σ(y, f(x)) = \begin{cases}
    \dfrac{1}{2}(y-f(x))^2    & |y-f(x)|≤σ
\\  σ(|y-f(x)|-\dfrac{1}{2}σ) & otherwise
\end{cases}$$

### Norm

$$\|x\|_{p} = \sqrt[p]{ \sum_{i=1}^{n} {|x_i|^{p}} }$$

- $\mathrm{MAE}(y, f(x)) = \dfrac{\|y - f(x)\|_1}{n}$

## 分类问题

### Cross Entropy

$$\mathrm{CE}(p, q) = -\sum\limits_{x∈X} p(x)\log q(x)$$

```python
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
```
