#!/usr/bin/python3

import numpy as np
from Activation_Functions import ActivationFunction
class Layer() :
    def __init__(self, num_nodes, inputs, activation_function) :
        self.weights = np.random.rand(num_nodes, inputs)
        self.biases = np.random.rand(num_nodes, 1)
        self.outputs = np.random.rand(num_nodes, 1)
        self.deltas = np.random.rand(num_nodes, 1)
        self.activation_function = activation_function

    def feed(self, input_array) :
        dot_product = np.dot(self.weights, input_array.T) + self.biases
        return ActivationFunction(dot_product,activation_function)
    
    def back_prop(self,outputs):
        pass