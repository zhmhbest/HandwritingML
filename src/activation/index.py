import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def leaky_relu(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return np.where(x >= 0, x, x * alpha)


def elu(x: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    return np.where(x >= 0, x, (alpha * (np.exp(x) - 1)))


def softmax(x: np.ndarray) -> np.ndarray:
    es = np.exp(x)
    return es / np.sum(es, axis=1, keepdims=True)


def stable_softmax(x: np.ndarray) -> np.ndarray:
    es = np.exp(x - np.max(x))
    return es / np.sum(es, axis=1, keepdims=True)


def distill_softmax(x: np.ndarray, t: int = 2) -> np.ndarray:
    es = np.exp(x / t)
    return es / np.sum(es, axis=1, keepdims=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure(figsize=[9.6, 7.4])

    plt.subplot(211)
    inputs = np.linspace(-10, 10).reshape(1, -1)
    plt.plot(inputs[0], 8 + sigmoid(inputs)[0], label='Sigmoid (+8)')
    plt.plot(inputs[0], 6 + tanh(inputs)[0], label='Tanh (+6)')
    plt.plot(inputs[0], 4 + relu(inputs)[0], label='Relu (+4)')
    plt.plot(inputs[0], 2 + leaky_relu(inputs)[0], label='Leaky Relu (+2)')
    plt.plot(inputs[0], 0 + elu(inputs)[0], label='Elu (+0)')
    plt.grid()
    plt.legend()
    # plt.show()

    plt.subplot(212)
    inputs = np.arange(10).reshape(1, -1)
    plt.plot(inputs[0], 1 + softmax(inputs)[0] * 10, label='Softmax (+1)')
    plt.plot(inputs[0], 2 + stable_softmax(inputs)[0] * 10, label='Stable Softmax (+2)')
    plt.plot(inputs[0], 3 + distill_softmax(inputs, 2)[0] * 10, label='Distill Softmax T=2 (+3)')
    plt.plot(inputs[0], 4 + distill_softmax(inputs, 3)[0] * 10, label='Distill Softmax T=3 (+4)')
    plt.grid()
    plt.legend()
    plt.show()
