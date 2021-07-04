import numpy as np
from numpy.core._multiarray_umath import ndarray

from frame.Layer import Layer
from initializers import ParametersInitializer, ActivationInitializer


# from torch.nn import Linear


class Linear(Layer):
    def __str__(self):
        return Linear.__name__

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation_name: str = "",
            initializer: str = ""
    ):
        """
        :param in_features: 输入尺寸
        :param out_features: 输出尺寸
        :param bias: 是否有偏执
        :param activation_name: 激活函数名称
        :param initializer: 参数初始化方法
        """
        super(Linear, self).__init__()
        initializer_activation = ActivationInitializer(activation_name)
        initializer_parameters = ParametersInitializer(initializer, str(activation_name))

        self._init_params(
            hyper_parameters={
                'in_features': in_features,
                'out_features': out_features,
                'bias': bias,
                'activation_name': activation_name,
                'activation': initializer_activation(),
                'initializer': initializer
            },
            parameters={
                "w": initializer_parameters((in_features, out_features)),
                "b": np.zeros((1, out_features)) if bias else None
            },
            derived_variables={
                'Y1': [],
                'Y2': []
            }
        )

    def forward(self, x: ndarray, retain_derived=True):
        """
        :param x:
        :param retain_derived: 是否保留forward时计算的中间变量
        :return: y ndarray
        """
        assert self.is_initialized, "Uninitialized Layer"
        w = self.parameters["w"]
        b = self.parameters["b"]
        y1 = x @ w + b
        y2 = self['activation'](y1)
        if retain_derived:
            self.derived_variables['Y1'].append(y1)
            self.derived_variables["Y2"].append(y2)
        return y2

    def backward(self, out: ndarray, **kwargs):
        pass

    # def backward(self, dLdy, retain_grads=True):
    #     """
    #     Backprop from layer outputs to inputs.
    #
    #     Parameters
    #     ----------
    #     dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)` or list of arrays
    #         The gradient(s) of the loss wrt. the layer output(s).
    #     retain_grads : bool
    #         Whether to include the intermediate parameter gradients computed
    #         during the backward pass in the final parameter update. Default is
    #         True.
    #
    #     Returns
    #     -------
    #     dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)` or list of arrays
    #         The gradient of the loss wrt. the layer input(s) `X`.
    #     """  # noqa: E501
    #     assert self.trainable, "Layer is frozen"
    #     if not isinstance(dLdy, list):
    #         dLdy = [dLdy]
    #
    #     dX = []
    #     X = self.X
    #     for dy, x in zip(dLdy, X):
    #         dx, dw, db = self._bwd(dy, x)
    #         dX.append(dx)
    #
    #         if retain_grads:
    #             self.gradients["W"] += dw
    #             self.gradients["b"] += db
    #
    #     return dX[0] if len(X) == 1 else dX
    #
    # def _bwd(self, dLdy, X):
    #     """Actual computation of gradient of the loss wrt. X, W, and b"""
    #     W = self.parameters["W"]
    #     b = self.parameters["b"]
    #
    #     Z = X @ W + b
    #     dZ = dLdy * self.act_fn.grad(Z)
    #
    #     dX = dZ @ W.T
    #     dW = X.T @ dZ
    #     dB = dZ.sum(axis=0, keepdims=True)
    #     return dX, dW, dB
    #
    # def _bwd2(self, dLdy, X, dLdy_bwd):
    #     """Compute second derivatives / deriv. of loss wrt. dX, dW, and db"""
    #     W = self.parameters["W"]
    #     b = self.parameters["b"]
    #
    #     dZ = self.act_fn.grad(X @ W + b)
    #     ddZ = self.act_fn.grad2(X @ W + b)
    #
    #     ddX = dLdy @ W * dZ
    #     ddW = dLdy.T @ (dLdy_bwd * dZ)
    #     ddB = np.sum(dLdy @ W * dLdy_bwd * ddZ, axis=0, keepdims=True)
    #     return ddX, ddW, ddB


if __name__ == '__main__':
    x = np.random.rand(1, 2)
    linear = Linear(2, 3)
    y = linear(x)

    print(y)

