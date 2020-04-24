#!/usr/bin/python3

import numpy as np

class ActivationFunction(object) :
    def __init__(self, Z, function_name) :
        self.Z = Z
        self.function = function_name
        self.activate()

    def activate(self) :
        if self.activation_function == 'sigmoid' :
            return self._sigmoid(self.Z)
        elif self.activation_function == 'tanh' :
            return self._tanh(self.Z)
        elif self.activation_function == 'relu':
            return self._relu(self.Z)
        elif self.activation_function == 'softmax':
            return self._softmax(self.Z)

    def _sigmoid(self, x) :
        return 1/(1 + np.exp(-x))
    def _tanh(self, x) :
        return 2/(1 + np.exp(-2*x)) -1
    def _relu(self, x):
        return np.max(0,x)
    def _softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))
