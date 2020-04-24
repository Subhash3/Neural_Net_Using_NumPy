#!/usr/bin/python3

import numpy as np

class Layer() :
    def __init__(self, num_nodes, inputs, activation_function) :
        self.weights = np.random.rand(num_nodes, inputs)
        self.biases = np.random.rand(num_nodes, 1)
        self.outputs = np.random.rand(num_nodes, 1)
        self.deltas = np.random.rand(num_nodes, 1)
        self.activation_function = activation_function

    def feed(self, input_array) :
        dot_product = np.dot(self.weights, np.transpose(input_array))
        dot_product += self.biases
        return self.activate(dot_product)

    def activate(self, x) :
        if self.activation_function == 'sigmoid' :
            return self._sigmoid(x)
        if self.activation_function == 'tanh' :
            return self._tanh(x)
        if self.activation_function == 'relu':
            return self._relu(x)
        if self.activation_function == 'softmax':
            return self._softmax(x)

    def _sigmoid(self, x) :
        return 1/(1 + np.exp(-x))

    def _tanh(self, x) :
        return 2/(1 + np.exp(-2*x)) -1
    def _relu(self, x):
        return np.max(0,x)
    def _softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))
