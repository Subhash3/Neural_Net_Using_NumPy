#!/usr/bin/python3

import numpy as np
from Layer import Layer

class NeuralNetwork() :
    def __init__(self, I, O) :
        self.Network = list()
        self.I = I
        self.O = O
    
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
        all_outputs = list()
        for layer in self.Network :
            outputs = layer.feed(input_array)
            all_outputs.append(outputs)
            input_array = outputs
        
        return outputs
