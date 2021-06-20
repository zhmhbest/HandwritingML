from numpy import ndarray
import numpy as np
from layers.base import LayerBase


class Softmax(LayerBase):
    def __init__(self, temperature: int = 1, dim: int = -1):
        super(Softmax, self).__init__(layer_name='Softmax')
        super(Softmax, self)._init_params(hyper_parameters={
            'temperature': temperature,
            'dim': dim,
        })

    def forward(self, x: ndarray, retain_derived: bool = True):
        print(self['dim'])
        # Distill Stable Softmax
        es = np.exp((x - np.max(x, axis=self['dim'], keepdims=True)) / self['temperature'])
        result = es / np.sum(es, axis=self['dim'], keepdims=True)

        if retain_derived:
            self.derived_variables['X'].append(x)
        return result

    def backward(self, out: ndarray, **kwargs):
        pass
