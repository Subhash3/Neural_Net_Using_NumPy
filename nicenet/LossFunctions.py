import numpy as np


class LossFunctions:
    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.function = self._mse

        if loss_function == 'mse':
            self.function = self._mse
        elif loss_function == 'ce':
            self.function = self._cross_entropy

    def get_loss(self, outputs, targets):
        # print("Outputs:", outputs)
        # print("Targets:", targets)
        loss = self.function(outputs, targets)
        # print("loss:", loss)
        # print()
        return loss

    def get_gradient(self, outputs, targets):
        # print("Outputs:", outputs)
        # print("Targets:", targets)
        gradient = self.function(outputs, targets, derivative=True)
        # print("gradient:", gradient)
        # print()
        return gradient

    def _mse(self, outputs, targets, derivative=False):
        # print("hihi")
        error = np.subtract(targets, outputs)
        if derivative:
            return -1*error
        return np.mean(error*error)

    def _cross_entropy(self, outputs, targets, derivative=False):
        error = np.subtract(targets, outputs)
        if derivative:
            return -1*error
        # print("cross_entropy:", targets*np.log(outputs))
        # print("cross_entropy:", np.sum(targets*np.log(outputs)))
        return -1*np.sum(targets*np.log(outputs))
