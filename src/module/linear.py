import numpy as np
from .base import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # self.weight = Parameter(torch.Tensor(out_features, in_features))
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

    def forward(self, **kwargs):
        super().forward(**kwargs)

    def backward(self, **kwargs):
        super().backward(**kwargs)
