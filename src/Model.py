#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import json

np.set_printoptions(precision=20)

## Class for activation functions
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

    def _softmax(self, x, derivative=False) :
        if derivative :
            return x * (1 - x)
        # np.exp(x)/np.sum(np.exp(x))
        exponentials = np.exp(x)
        total = np.sum(exponentials)
        return exponentials/total

## Layer class
class Layer() :
    def __init__(self, num_nodes, inputs, activation_function) :
        self.inputs = inputs
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
        print("\tInputs: ", self.inputs)
        print("\tWeights: ", self.weights)
        print("\tBiases: ", self.biases.T)
        print("\tDeltas: ", self.deltas.T)
        print("\tOutputs: ", self.outputs.T)
        print("\tActivation: ", self.activation_function)

## Main NeuralNetwork class
class NeuralNetwork() :
    def __init__(self, I, O,cost='mse') :
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

        Returns
        -------
        Doesn't return anything
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
        if self.model_compiled :
            print("[!!] Model is already compiled!")
            print("[!!] You cannot add layers anymore")
            return
        self.isLoadedModel = False
        self.model_compiled = True
        # Adding output layer
        self.addLayer(self.O, activation_function=activation_function)
    
    def feedforward(self, input_array) :
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

        if not self.model_compiled :
            print("[-] Network is not complete.!")
            print("[!!] Please compile the network by adding an output layer")
            return

        self.all_errors = list()
        self.epochs = epochs
        for epoch in range(epochs) :
            self.MSE = 0
            for i in range(size) :
                data_sample = Dataset[i]
                # input_array = data_sample[0]
                # target_array = data_sample[1]
                input_array = data_sample[0]
                target_array = data_sample[1]

                all_outputs = self.feedforward(input_array)
                output_error = self.backpropagate(target_array)
                self.update_weights(input_array)
                
                self.MSE += output_error
                if logging :
                    print(input_array.transpose(), "\x1b[35m", all_outputs, "\x1b[0m", target_array, "\x1b[31m", output_error, "\x1b[0m")

            self.MSE /= size
            print("Epoch: ", epoch+1, " ==> Error: ", self.MSE)
            self.all_errors.append(self.MSE)
            self.accuracy =  (1 - np.sqrt(self.MSE))*100

            if logging :
                print()

    def predict(self, input_array) :
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

    def epoch_vs_error(self) :
        """
        Plot error vs epoch graph

        Parameters
        ----------
        Doesn't accept any parameters

        Returns
        -------
        Doesn't return anything
        """
        if self.isLoadedModel :
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

    def evaluate(self) :
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
        if self.accuracy < 70 :
            print("\t\tModel Doesn't seem to have fit the data correctly")
            print("\t\tTry increasing the epochs")
        print("\t=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        
    def display(self) :
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
        for i in range(self.total_layers) :
            print("Layer: ", i+1)
            self.Network[i].display()

    def export_model(self, filename) :
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
        try :
            fhand = open(filename, 'w')
        except Exception as e :
            print("[!!] Unable to open file ", filename, ": ", e)
            print("[!!] Couldn't export model")
            return

        model_info = dict()
        model_info["inputs"] = self.I
        model_info["outputs"] = self.O
        model_info["learning_rate"] = self.learningRate
        model_info["model_compiled"] = self.model_compiled
        model_info["layers"] = list()
        for layer in self.Network :
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
    def load_model(filename) :
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
        try :
            fhand = open(filename, 'r')
        except Exception as e :
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

        for layer_object in model_info["layers"] :
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

## Dataset class
class Dataset() :
    def __init__(self, I, O) :
        self.I = I
        self.O = O
        self.size = 0
        self.dataset =  list()

    def makeDataset(self, inputFile, targetFile) :
        input_handler = self.openFile(inputFile)
        target_handler = self.openFile(targetFile)

        if not input_handler or not target_handler :
            print("Unable to create Dataset")
            return
        input_lines = input_handler.readlines()
        target_lines = target_handler.readlines()

        for inp, tar in zip(input_lines, target_lines) :
            input_array = list(map(float, inp.split(',')))
            target_array = list(map(float, tar.split(',')))

            sample = list()
            sample.append(np.reshape(input_array, (self.I, 1)))
            sample.append(np.reshape(target_array, (self.O, 1)))
            self.dataset.append(sample)
            self.size += 1
    
    def modifyLists(self, input_array, target_array) :
        sample = list()
        sample.append(np.reshape(input_array, (self.I, 1)))
        sample.append(np.reshape(target_array, (self.O, 1)))
        self.dataset.append(sample)
        self.size += 1

    def getRawData(self) :
        return self.dataset, self.size
    
    def display(self) :
        for i in range(self.size) :
            sample = self.dataset[i]
            print("Data Sample:", i+1)
            print("\tInput: ", sample[0])
            print("\tTarget: ", sample[1])
    
    @staticmethod
    def openFile(filename) :
        try :
            fhand = open(filename)
        except Exception as e:
            print("[!!] Exception Occurred while Opening", filename, e)
            return None
        return fhand
    
    @staticmethod
    def scaleData(array, size, scale_range=(0, 1)) :
        if scale_range == (0, 1) :
            small = large = None
            for row in array :
                num = row[0]
                if small == None or num < small :
                    small = num
                if large == None or num > large :
                    large = num
            for i in range(size) :
                array[i] = (array[i] - small) / (large - small)