import numpy as np
from .ActivationFunction import ActivationFunction
from .LossFunctions import LossFunctions

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

    def feed(self, input_array):
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
        return self.activator.activate(x)

    def calculate_gradients(self, target_or_weights, layer_type, next_layer_deltas=None):
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

    def update_weights(self, inputs, learningRate):
        change_in_weights = np.dot(self.deltas, inputs.T) * learningRate
        self.weights = np.subtract(self.weights, change_in_weights)
        self.biases = np.subtract(self.biases, self.deltas * learningRate)

    def display(self):
        print("\tInputs: ", self.inputs)
        print("\tWeights: ", self.weights)
        print("\tBiases: ", self.biases.T)
        print("\tDeltas: ", self.deltas.T)
        print("\toutput_array: ", self.output_array.T)
        print("\tActivation: ", self.activation_function)
