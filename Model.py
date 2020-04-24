#!/usr/bin/python3

import numpy as np
from Layer import Layer

class NeuralNetwork() :
    def __init__(self, I, O,cost='mse') :
        self.Network = list()
        self.I = I
        self.O = O
        self.cost = cost
        self.error = 0
    
    def addLayer(self, num_nodes, activation_function="sigmoid") :
        """
        Adds a layer to the network.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the hidden layer

        [activation_function] :str
            It is an optional parameter.
            Specifies the activation function of the layer.
            Default value is sigmoid.
        """
        # A layer can be thought of as a matrix
        # No. of row = no. of nodes
        # No. of columns = No. of weights = No. of inputs + 1 (bias)
        layer = Layer(num_nodes, self.I, activation_function)
        self.Network.append(layer)

    def addOutputLayer(self, activation_function="sigmoid") :
        # Adding output layer
        self.addLayer(self.O, activation_function=activation_function)
    
    def feedforward(self, input_array) :
        self.all_outputs = list()
        for layer in self.Network :
            outputs = layer.feed(input_array)
            all_outputs.append(outputs)
            input_array = outputs
        return all_outputs
    
    def cost_compute(self,Y):
        if self.cost == 'mse':
            self.error = np.sqrt((self.all_outputs - Y)**2)

    def back_propagate(self):
        self.all_grads = list()
        for i in range(len(self.Network),-1,-1):
            grad = 1
