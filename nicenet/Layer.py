import numpy as np
from .ActivationFunction import ActivationFunction
from .LossFunctions import LossFunctions
from .Types import T_Feature_Array, T_Output_Array

# Layer class


class Layer:
    def __init__(self, num_nodes, inputs, activation_function, loss_function):
        """
            Layer constructor

            Parameters
            ----------
            num_nodes : int
                No. of nodes in the layer

            inputs : int
                No. of inputs to the layer

            activation_function

            Returns
            -------
            None
        """
        self.inputs = inputs
        self.num_nodes = num_nodes
        self.weights = np.random.randn(
            num_nodes, inputs) * np.sqrt(2 / num_nodes)
        self.biases = np.random.randn(num_nodes, 1)
        self.output_array = np.random.randn(num_nodes, 1)
        self.deltas = np.zeros((num_nodes, 1))
        self.activation_function = activation_function
        self.activator = ActivationFunction(self.activation_function)
        self.loss_function = loss_function
        self.loss_computer = LossFunctions(self.loss_function)

    def feed(self, input_array: T_Feature_Array) -> T_Output_Array:
        """
            Feeds the given input array to a particular layer.

            Parameters
            ----------
            input_array: T_Feature_Array
                Input array to be fed to the layer

            Returns
            -------
            output_array: T_Output_Array
        """
        # print("Weights", self.weights.shape)
        # print("Inputs ", input_array.shape)
        self.input_array = input_array
        dot_product = np.dot(self.weights, input_array)
        dot_product += self.biases
        # print("Output: ", dot_product.shape)
        output_array = self.activate(dot_product)
        self.output_array = output_array  # Store the output in the layer.
        # print(self.input_array.shape, self.output_array.shape)
        return output_array

    def activate(self, x):
        """
            Passes the output array to an activation function.

            Parameters
            ----------
            x
                Output array from a layer

            Returns
            -------
            Activated output
        """
        return self.activator.activate(x)

    def calculate_gradients(self, target_or_weights, layer_type, next_layer_deltas=None):
        """
            Calculates the gradients for each weight and bias

            Parameters
            ----------
            target_or_weights
                This is either targers array of weights matrix.
                Specifically, it'll be the targets array while computing the gradients for the output layer
                and weights matrix of the next layer.

            layer_type
                This will either be "hidden" or "output"

            [next_layer_deltas]
                This is (not exactly) an optional parameter.
                This will be passed only while computing the gradients of a hidden layer.

            Returns
            -------
                Doesn't return anything. But stores the gradients as a class attribute.
        """

        activation_gradient = self.activator.activate(
            self.output_array, derivative=True)
        if layer_type == "output":
            # print("Output Layer")
            # target_or_weights => target values
            loss_gradient = self.loss_computer.get_gradient(
                self.output_array, target_or_weights)
            self.deltas = loss_gradient * activation_gradient

            # delta = (target - output) * activation_derivative(output)
        else:
            # print("Hidden Layer")
            # target_or_weights = Next layer's weights
            hidden_errors = np.dot(target_or_weights.transpose(
            ), next_layer_deltas)  # Errors in the hidden layer
            self.deltas = hidden_errors * activation_gradient
            # deltas += (next_layers weights * next_layers deltas)

    def update_weights(self, inputs, learning_rate):
        """
            Tweak the weights of the layer.

            Parameters
            ----------
            inputs: T_Feature_Array
                Input to this network

            learning_rate: float
                Learning rate of the entire network.

            Returns
            -------
            Doesn't return anything.
        """

        change_in_weights = np.dot(self.deltas, inputs.T) * learning_rate
        self.weights = np.subtract(self.weights, change_in_weights)
        self.biases = np.subtract(self.biases, self.deltas * learning_rate)

    def display(self):
        """
            Display the metadata of the layer.
        """

        print("\tInputs: ", self.inputs)
        print("\tWeights: ", self.weights)
        print("\tBiases: ", self.biases.T)
        print("\tDeltas: ", self.deltas.T)
        print("\toutput_array: ", self.output_array.T)
        print("\tActivation: ", self.activation_function)
