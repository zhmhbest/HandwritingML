from copy import deepcopy


class HyperParametersHolder:
    def __init__(self, **kwargs):
        self.hyper_parameters: dict = kwargs

    def __setitem__(self, key, value):
        self.hyper_parameters[key] = value

    def __getitem__(self, key):
        return self.hyper_parameters[key]

    def copy(self):
        return deepcopy(self)

    def get_hyper_parameters(self) -> dict:
        return self.hyper_parameters
