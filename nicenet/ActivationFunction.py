from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")

# Class for activation functions


class ActivationFunction(Generic[T]):
    def __init__(self, function_name):
        if function_name == "sigmoid":
            self.activation_function = self._sigmoid
        elif function_name == "tanh":
            self.activation_function = self._tanh
        elif function_name == "relu":
            self.activation_function = self._relu
        elif function_name == "softmax":
            self.activation_function = self._softmax
        elif function_name == "identity":
            self.activation_function = self._identity

    def activate(self, x: T, derivative=False) -> T:
        if derivative:
            return self.activation_function(x, derivative=True)
        return self.activation_function(x)

    def _sigmoid(self, x: T, derivative=False) -> T:
        if derivative:
            # y = self._sigmoid(x)
            y = x
            return y * (1 - y)
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x: T, derivative=False) -> T:
        tan = 2 / (1 + np.exp(-2 * x)) - 1
        if derivative:
            return 1 - tan ** 2
        return tan

    def _relu(self, x: T, derivative=False) -> T:
        if derivative:
            return (x > 0).astype(int)
        return np.max(0, x)

    def _softmax(self, x: T, derivative=False) -> T:
        if derivative:
            return x * (1 - x)
        # np.exp(x)/np.sum(np.exp(x))
        exponentials = np.exp(x)
        total = np.sum(exponentials)
        return exponentials / total

    def _identity(self, x: T, derivative=False) -> T:
        if derivative:
            return np.ones_like(x)
        return x
