#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from .Layer import Layer
from .Dataset import Dataset
import json
import time

np.set_printoptions(precision=20)


class NeuralNetwork():
    # Main NeuralNetwork class
    def __init__(self, I, O, cost='mse'):
        """
        Creates a Feed Forward Neural Network.

        Parameters
        ----------
        I : int
            Number of inputs to the network

        O : int
            Number of outputs from the network

        Returns
        -------
        Doesn't return anything
        """
        self.Network = list()
        self.I = I
        self.O = O
        self.cost = cost
        self.total_layers = 0
        self.learningRate = 0.5
        self.isLoadedModel = False
        self.model_compiled = False

    def setLearningRate(self, lr):
        self.learningRate = lr

    def addLayer(self, num_nodes, activation_function="sigmoid"):
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

        Returns
        -------
        Doesn't return anything
        """
        # A layer can be thought of as a matrix
        # No. of row = no. of nodes
        # No. of columns = No. of weights = No. of inputs + 1 (bias)
        if self.total_layers == 0:
            inputs = self.I
        else:
            last_layer = self.Network[-1]
            inputs = last_layer.num_nodes
        layer = Layer(num_nodes, inputs, activation_function)
        self.Network.append(layer)
        self.total_layers += 1

    def compile(self, activation_function="sigmoid"):
        """
        Basically, it just adds the output layer to the network.

        Parameters
        ----------
        [activation_function] :str
            It is an optional parameter.
            Specifies the activation function of the layer.
            Default value is sigmoid.

        Returns
        -------
        Doesn't return anything
        """
        if self.model_compiled:
            print("[!!] Model is already compiled!")
            print("[!!] You cannot add layers anymore")
            return
        self.isLoadedModel = False
        self.model_compiled = True
        # Adding output layer
        self.addLayer(self.O, activation_function=activation_function)

    def feedforward(self, input_array):
        """
        Feeds the given input throughout the network

        Parameters
        ----------
        input_array : np.array()
            It is columnar vector of size Inputs x 1
            It is the input fed to the network

        Returns
        -------
        all_outputs : np.array()
            An array of all the outputs produced by each layer.
        """
        all_outputs = list()
        _i = 1
        for layer in self.Network:
            # print("Feeding ", input_array.T, "to , layer", i)
            outputs = layer.feed(input_array)
            all_outputs.append(outputs.T)
            input_array = outputs
            # print("All outputs: ", all_outputs)
            # print()
            # i += 1
        return all_outputs

    def backpropagate(self, target):
        """
        Backpropagate the error throughout the network
        This function is called inside the model only.

        Parameters
        ----------
        target : np.array()
            It is columnar vector of size Outputs x 1
            It is the ground truth value corresponding to the input

        Returns
        -------
        Error : float
            Returns the Mean Squared Error of the particular output
        """
        for i in range(self.total_layers-1, -1, -1):
            layer = self.Network[i]
            if i == self.total_layers - 1:
                # print("Output layer: ", layer.outputs, "Target: ", target)
                output_errors = (target - layer.outputs)**2
                # print("Error: ", output_errors)
                layer.calculate_gradients(target, "output")
            else:
                next_layer = self.Network[i+1]
                layer.calculate_gradients(
                    next_layer.weights, "hidden", next_layer.deltas)
        return sum(output_errors)

    def update_weights(self, input_array):
        """
        Update the weights of the network.
        This function is called inside the model only.

        Parameters
        ----------
        input_array : np.array()
            It is columnar vector of size Inputs x 1
            It is the input fed to the network

        Returns
        -------
        Doesn't return anything
        """
        for i in range(self.total_layers-1, -1, -1):
            layer = self.Network[i]
            if i == 0:
                # if it is the first layer => inputs = input_array
                layer.update_weights(input_array, self.learningRate)
            else:
                # not the first most => inputs = previous layer's output
                inputs = self.Network[i-1].outputs
                layer.update_weights(inputs, self.learningRate)

    def Train(self, Dataset, size, epochs=5000, logging=True):
        """
        Trains the neural network using the given dataset.

        Parameters
        ----------
        Dataset : Dataset() object
            It is a dataset object contains the dataset.

        size : int
            Size of the dataset

        [epochs] : int
            An optional parameter.
            Number of epochs to train the network. Default value is 5000

        [logging] : bool
            An optional parameter.
            If its true, all outputs from the network will be logged out onto STDOUT for each epoch.

        Returns
        -------
        Doesn't return anything.
        """

        if not self.model_compiled:
            print("[-] Network is not complete.!")
            print("[!!] Please compile the network by adding an output layer")
            return

        self.all_errors = list()
        self.epochs = epochs
        for epoch in range(epochs):
            self.MSE = 0
            # One Epoch
            if epoch == 0:
                start = time.time()
            elif epoch == 1:
                end = time.time()
                seconds = (end-start)*epochs

                minutes = seconds//60
                seconds = seconds % 60

                hours = minutes // 60
                minutes = minutes % 60
                print("Estimated Training Time: ", hours, "hrs ",
                      minutes, "min ", seconds, "sec", sep="")

            for i in range(size):
                data_sample = Dataset[i]
                # input_array = data_sample[0]
                # target_array = data_sample[1]
                input_array = data_sample[0]
                target_array = data_sample[1]

                all_outputs = self.feedforward(input_array)
                output_error = self.backpropagate(target_array)
                self.update_weights(input_array)

                self.MSE += output_error
                if logging:
                    print(input_array.transpose(
                    ), "\x1b[35m", all_outputs, "\x1b[0m", target_array, "\x1b[31m", output_error, "\x1b[0m")

            self.MSE /= size
            print("Epoch: ", epoch+1, " ==> Error: ", self.MSE)
            self.all_errors.append(self.MSE)
            self.accuracy = (1 - np.sqrt(self.MSE))*100

            if logging:
                print()

    def predict(self, input_array):
        """
        Predicts the output using the given input.

        Parameters
        ----------
        input_array : np.array()
            It is columnar vector of size Inputs x 1
            It is the input fed to the network

        Returns
        -------
        prediction : np.array()
            Predicted value produced by the network.
        """
        return self.feedforward(input_array)[-1]

    def epoch_vs_error(self):
        """
        Plot error vs epoch graph

        Parameters
        ----------
        Doesn't accept any parameters

        Returns
        -------
        Doesn't return anything
        """
        if self.isLoadedModel:
            print("[!!] You cannot look at epoch vs error graph in a loaded model")
            print("[!!] You can only look at that while training.!")
            print("[!!] Make some modifications to the network to own the model")
            return
        all_epochs = [i+1 for i in range(self.epochs)]
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Epoch vs Error")
        plt.plot(all_epochs, self.all_errors)
        plt.show()

    def evaluate(self):
        """
        Print the basic information about the network.
        Like accuracy, error ..etc.

        Parameters
        ----------
        Doesn't accept any parameters

        Returns
        -------
        Doesn't return anything
        """
        print("\t=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print("\tModel is trained for ", self.epochs, "epochs")
        print("\tModel Accuracy: ", self.accuracy, "%")
        if self.accuracy < 70:
            print("\t\tModel Doesn't seem to have fit the data correctly")
            print("\t\tTry increasing the epochs")
        print("\t=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

    def display(self):
        """
        Print the information of each layer of the network.
        It can be used to debug the network!

        Parameters
        ----------
        Doesn't accept any parameters

        Returns
        -------
        Doesn't return anything
        """
        for i in range(self.total_layers):
            print("Layer: ", i+1)
            self.Network[i].display()

    def export_model(self, filename):
        """
        Export the model to a json file

        Parameters
        ----------
        filename: str
            File name to export model

        Returns
        -------
        Doesn't return anything
        """
        try:
            fhand = open(filename, 'w')
        except Exception as e:
            print("[!!] Unable to open file ", filename, ": ", e)
            print("[!!] Couldn't export model")
            return

        model_info = dict()
        model_info["inputs"] = self.I
        model_info["outputs"] = self.O
        model_info["learning_rate"] = self.learningRate
        model_info["model_compiled"] = self.model_compiled
        model_info["layers"] = list()
        for layer in self.Network:
            layer_object = dict()
            layer_object["neurons"] = layer.num_nodes
            layer_object["inputs"] = layer.inputs
            layer_object["weights"] = layer.weights.tolist()
            layer_object["biases"] = layer.biases.tolist()
            layer_object["activation_function"] = layer.activation_function
            model_info["layers"].append(layer_object)
        model_info["accuracy"] = self.accuracy[0]
        model_info["MSE"] = self.MSE[0]
        model_info["epochs"] = self.epochs

        json_format_string = json.dumps(model_info)
        fhand.write(json_format_string)
        fhand.close()
        print("[*] Model exported successfully to", filename)

    @staticmethod
    def load_model(filename):
        """
        Load model from an eported (json) model

        Parameters
        ----------
        filename : str
            Exported model (json) file

        Returns
        -------
        brain : NeuralNetwork
            NeuralNetwork object
        """
        try:
            fhand = open(filename, 'r')
        except Exception as e:
            print("[!!] Unable to open file ", filename, ": ", e)
            print("[!!] Couldn't load model")
            return

        model_info = json.load(fhand)
        # print(model_info)

        inputs = model_info["inputs"]
        outputs = model_info["outputs"]

        brain = NeuralNetwork(inputs, outputs)
        brain.total_layers = 0
        brain.isLoadedModel = True

        for layer_object in model_info["layers"]:
            num_nodes = layer_object["neurons"]
            weights = layer_object["weights"]
            inputs = layer_object["inputs"]
            biases = layer_object["biases"]
            activation_fn = layer_object["activation_function"]

            layer = Layer(num_nodes, inputs, activation_function=activation_fn)
            layer.weights = np.array(weights)
            layer.biases = np.array(biases)
            brain.Network.append(layer)
            brain.total_layers += 1

        brain.accuracy = model_info["accuracy"]
        brain.MSE = model_info["MSE"]
        brain.epochs = model_info["epochs"]
        brain.learningRate = model_info["learning_rate"]
        brain.model_compiled = model_info["model_compiled"]
        print("[*] (", filename, ") Model Loaded successfully", sep="")

        return brain
