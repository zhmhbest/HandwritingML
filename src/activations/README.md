# Activation

[TOC]

## Sigmoid

$$
σ(x) = \dfrac{1}{1 + e^{-x}}
$$

### grad1

$$
\begin{array}{l}
    \dfrac{∂σ}{∂x} \\ \\\\ \\\\ \\\\
\end{array}
\begin{array}{l}
        = -(1+e^{-x})^{-2} \cdot (0 - e^{-x})
\\\\    = e^{-x} \cdot (1+e^{-x})^{-2}
\\\\    = ((1 + e^{-x}) -1) \cdot (1+e^{-x})^{-2}
\\\\    = σ(x) \cdot (1 - σ(x))
\end{array}
$$

### grad2

$$
\begin{array}{l}
    \dfrac{∂^2σ}{∂x^2} \\ \\\\ \\\\
\end{array}
\begin{array}{l}
        = \frac{∂σ}{∂x} \cdot (1 - σ(x)) + σ(x) \cdot (0 - \frac{∂σ}{∂x})
\\\\    = \frac{∂σ}{∂x} \cdot (1 - 2 σ(x))
\\\\    = σ(x) \cdot (1 - σ(x)) \cdot (1 - 2 σ(x))
\end{array}
$$

## Tanh

$$
f(x) = \dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

### grad1

$$
\begin{array}{l}
    \dfrac{∂f}{∂x} \\ \\\\
\end{array}
\begin{array}{l}
        = \big[{ (e^x + e^{-x})^2 - (e^x - e^{-x})^2 }\big] \cdot (e^x + e^{-x})^{-2}
\\\\    = 1 - f^2(x)
\end{array}
$$

### grad2

$$
\begin{array}{l}
    \dfrac{∂^2f}{∂x^2} \\ \\\\
\end{array}
\begin{array}{l}
        = -2f(x) \cdot \frac{∂f}{∂x}
\\\\    = 2\big(f^3(x) - f(x)\big)
\end{array}
$$

## ReLU

$$
f(x) = \max(x, 0)
$$

### grad1

$$
\dfrac{∂f}{∂x} =
\begin{cases}
    1 & x > 0
\\  0 & \text{otherwise}
\end{cases}
$$

### grad2

$$
\dfrac{∂^2f}{∂x^2} = 0
$$

## Leaky ReLU

$$
f(x) = \begin{cases}
        x   & x > 0
    \\  αx  & \text{otherwise}
\end{cases}
$$

### grad1

$$
\dfrac{∂f}{∂x} =
\begin{cases}
    1 & x > 0
\\  α & \text{otherwise}
\end{cases}
$$

### grad2

$$
\dfrac{∂^2f}{∂x^2} = 0
$$

## ELU

$$
f(x) = \begin{cases}
    x               & x>0
    \\  α(e^x-1)    & \text{otherwise}
\end{cases}
$$

- $α$：ELU负值部分在何时饱和。

### grad1

$$
\dfrac{∂f}{∂x} =
\begin{cases}
    1 & x > 0
\\  αe^x & \text{otherwise}
\end{cases}
$$

### grad2

$$
\dfrac{∂^2f}{∂x^2} =
\begin{cases}
    0 & x > 0
\\  αe^x & \text{otherwise}
\end{cases}
$$

## SoftPlus

$$
f(x) = \log(1 + e^x)
$$

### grad1

$$
\frac{∂f}{∂x} = \frac{e^x}{1 + e^x}
$$

### grad2

$$
\dfrac{∂^2f}{∂x^2} = \dfrac{e^x}{(1 + e^x)^2}
$$
