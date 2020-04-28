#!/usr/bin/python3

import numpy as np
from Layer import Layer
from matplotlib import pyplot as plt

np.set_printoptions(precision=20)

class NeuralNetwork() :
    def __init__(self, I, O,cost='mse') :
        self.Network = list()
        self.I = I
        self.O = O
        self.cost = cost
        self.error = 0
        self.total_layers = 0
        self.learningRate = 0.5
    
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
        if self.total_layers == 0 :
            inputs = self.I
        else :
            last_layer = self.Network[-1]
            inputs = last_layer.num_nodes
        layer = Layer(num_nodes, inputs, activation_function)
        self.Network.append(layer)
        self.total_layers += 1

    def compile(self, activation_function="sigmoid") :
        # Adding output layer
        self.addLayer(self.O, activation_function=activation_function)
    
    def feedforward(self, input_array) :
        all_outputs = list()
        i = 1
        for layer in self.Network :
            # print("Feeding ", input_array.T, "to , layer", i)
            outputs = layer.feed(input_array)
            all_outputs.append(outputs.T)
            input_array = outputs
            # print("All outputs: ", all_outputs)
            # print()
            # i += 1
        return all_outputs

    def backpropagate(self, target) :
        for i in range(self.total_layers-1, -1, -1) :
            layer = self.Network[i]
            if i == self.total_layers -1 :
                # print("Output layer: ", layer.outputs, "Target: ", target)
                output_errors = (target - layer.outputs)**2
                # print("Error: ", output_errors)
                layer.calculate_gradients(target, "output")
            else :
                next_layer = self.Network[i+1]
                layer.calculate_gradients(next_layer.weights, "hidden", next_layer.deltas)
        return sum(output_errors)

    def update_weights(self, input_array) :
        for i in range(self.total_layers-1, -1, -1) :
            layer = self.Network[i]
            if i == 0 :
                # if it is the first layer => inputs = input_array
                layer.update_weights(input_array, self.learningRate)
            else :
                # not the first most => inputs = previous layer's output
                inputs = self.Network[i-1].outputs
                layer.update_weights(inputs, self.learningRate)

    def Train(self, Dataset, size, epochs=5000, logging=True) :
        self.all_errors = list()
        self.epochs = epochs
        for epoch in range(epochs) :
            self.MSE = 0
            for i in range(size) :
                data_sample = Dataset[i]
                # input_array = data_sample[0]
                # target_array = data_sample[1]
                input_array = data_sample.input_array
                target_array = data_sample.targets

                all_outputs = self.feedforward(input_array)
                output_error = self.backpropagate(target_array)
                self.update_weights(input_array)
                
                self.MSE += output_error
                if logging or epoch == epochs-1:
                    print(input_array.transpose(), "\x1b[35m", all_outputs, "\x1b[0m", target_array, "\x1b[31m", output_error, "\x1b[0m")

            self.MSE /= size
            print("Epoch: ", epoch+1, " ==> Error: ", self.MSE)
            self.all_errors.append(self.MSE)
            if logging :
                print()
    def predict(self, input_array) :
        return self.feedforward(input_array)

    def epoch_vs_error(self) :
        all_epochs = [i+1 for i in range(self.epochs)]
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Epoch vs Error")
        plt.plot(all_epochs, self.all_errors)
        plt.show()

    def evaluate(self) :
        self.accuracy =  (1 - np.sqrt(self.MSE))*100
        print("\t=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print("\tModel is trained for ", self.epochs, "epochs")
        print("\tModel Accuracy: ", self.accuracy, "%")
        if self.accuracy < 70 :
            print("\t\tModel Doesn't seem to have fit the data correctly")
            print("\t\tTry increasing the epochs")
        print("\t=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        
    def display(self) :
        for i in range(self.total_layers) :
            print("Layer: ", i+1)
            self.Network[i].display()
