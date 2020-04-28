#!/usr/bin/python3

import numpy as np
from Activation_Function import ActivationFunction

np.set_printoptions(precision=20)

class Layer() :
    def __init__(self, num_nodes, inputs, activation_function) :
        self.num_nodes = num_nodes
        self.weights = np.random.rand(num_nodes, inputs)
        self.biases = np.random.rand(num_nodes, 1)
        self.outputs = np.random.rand(num_nodes, 1)
        self.deltas = np.zeros((num_nodes, 1))
        self.activation_function = activation_function

    def feed(self, input_array) :
        # print("Weights", self.weights.shape)
        # print("Inputs ", input_array.shape)
        dot_product = np.dot(self.weights, input_array)
        dot_product += self.biases
        # print("Output: ", dot_product.shape)
        outputs = self.activate(dot_product)
        self.outputs = outputs # Store the output in the layer.
        return outputs

    def activate(self, x) :
        activator = ActivationFunction(self.activation_function)
        return activator.activate(x)

    def calculate_gradients(self, target_or_weights, layer_type, next_layer_deltas=None) :
        activator = ActivationFunction(self.activation_function)
        if layer_type == "output" :
            # print("Output Layer")
            # target_or_weights = target values
            self.deltas = (target_or_weights - self.outputs) * activator.activate(self.outputs, derivative=True)
            # delta = (target - output) * activation_derivative(output)
        else :
            # print("Hidden Layer")
            # target_or_weights = Next layer's weights
            hidden_errors = np.dot(target_or_weights.transpose(), next_layer_deltas) # Errors in the hidden layer
            self.deltas = hidden_errors * activator.activate(self.outputs, derivative=True)
            # deltas += (next_layers weights * next_layers deltas)
    
    def update_weights(self, inputs, learningRate) :
        change_in_weights = np.dot(self.deltas, inputs.T) * learningRate
        self.weights = np.add(self.weights, change_in_weights)
        self.biases += (self.deltas * learningRate)

    def display(self) :
        print("\tWeights: ", self.weights)
        print("\tBiases: ", self.biases)
        print("\tDeltas: ", self.deltas)
        print("\tOutputs: ", self.outputs)
        print("\tActivation: ", self.activation_function)
