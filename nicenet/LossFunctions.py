import numpy as np


class LossFunctions:
    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.function = self._mse

        if loss_function == 'mse':
            self.function = self._mse

    def get_loss(self, outputs, targets):
        return self.function(outputs, targets)

    def get_gradient(self, outputs, targets):
        return self.function(outputs, targets, derivative=True)

    def _mse(self, outputs, targets, derivative=False):
        error = np.subtract(targets, outputs)
        if derivative:
            return error
        return error*error
