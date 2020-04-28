#!/usr/bin/python3

import numpy as np

class ActivationFunction() :
    def __init__(self, function_name) :
        if function_name == 'sigmoid' :
            self.activation_function = self._sigmoid
        elif function_name == 'tanh' :
            self.activation_function = self._tanh
        elif function_name == 'relu':
            self.activation_function = self._relu
        elif function_name == 'softmax':
            self.activation_function = self._softmax

    def activate(self, x, derivative=False) :
        if derivative :
            return self.activation_function(x, derivative=True)
        return self.activation_function(x)

    def _sigmoid(self, x, derivative=False) :
        if derivative :
            return x * (1 - x)
        return 1/(1 + np.exp(-x))

    def _tanh(self, x, derivative=False) :
        tan = 2/(1 + np.exp(-2*x)) -1
        if derivative:
            return (1-tan**2)
        return tan

    def _relu(self, x, derivative=False) :
        if derivative:
            return (x>0).astype(int)
        return np.max(0,x)

    def _softmax(self, x) :
        return np.exp(x)/np.sum(np.exp(x))