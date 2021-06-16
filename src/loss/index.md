# Loss

[TOC]

## 回归问题

### MAE

$$
\mathrm{MAE}(y, f(x)) = \dfrac{\sum\nolimits_{i=1}^{n}|y_i-f(x_i)|}{n}
$$

### MSE

$$
\mathrm{MSE}(y, f(x)) = \dfrac{\sum\nolimits_{i=1}^{n}(y_i-f(x_i))^2}{n}
$$

### RMSE

$$
\mathrm{RMSE}(y, f(x)) = \sqrt{ \dfrac{\sum\nolimits_{i=1}^{n}(y_i-f(x_i))^2}{n}}
$$

### Huber

>建议$δ=1.35$以达到$95\%$的有效性。

$$
L_σ(y, f(x)) = \begin{cases}
        \dfrac{1}{2}(y-f(x))^2    & |y-f(x)|≤σ
    \\  σ(|y-f(x)|-\dfrac{1}{2}σ) & otherwise
\end{cases}
$$

### Norm

$$
\|x\|_{p} = \sqrt[p]{ \sum_{i=1}^{n} {|x_i|^{p}} }
$$

- $\mathrm{MAE}(y, f(x)) = \dfrac{\|y - f(x)\|_1}{n}$

## 分类问题

### Cross Entropy

$$
\mathrm{CE}(p, q) = -\sum\limits_{x∈X} p(x)\log q(x)
$$
