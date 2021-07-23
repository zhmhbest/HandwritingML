from numpy import ndarray

from hwml.frame.model import Model, ParameterLayer


class Sequential(Model):
    def train(self, x_train: ndarray, y_train: ndarray):
        assert self.optimizer is not None
        assert self.loss is not None

        self.optimizer.step()

        # forward
        y_pred = self.forward(x_train)
        loss_val = self.loss(y_train, y_pred)

        # backward
        g = self.loss.gradients
        for i in range(len(self.layers) - 1, -1, -1):
            layer: ParameterLayer = self.layers[i]
            g = layer.backward(g)

        self.loss.clear_gradients()
        return loss_val
