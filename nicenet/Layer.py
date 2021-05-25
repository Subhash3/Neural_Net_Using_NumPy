import numpy as np
from .ActivationFunction import ActivationFunction


# Layer class
class Layer:
    def __init__(self, num_nodes, inputs, activation_function):
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
        self.weights = np.random.randn(num_nodes, inputs) * np.sqrt(2 / num_nodes)
        self.biases = np.random.randn(num_nodes, 1)
        self.output_array = np.random.randn(num_nodes, 1)
        self.deltas = np.zeros((num_nodes, 1))
        self.activation_function = activation_function
        self.activator = ActivationFunction(self.activation_function)

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
        if layer_type == "output":
            # print("Output Layer")
            # target_or_weights => target values
            self.deltas = (
                target_or_weights - self.output_array
            ) * self.activator.activate(self.output_array, derivative=True)
            # delta = (target - output) * activation_derivative(output)
        else:
            # print("Hidden Layer")
            # target_or_weights = Next layer's weights
            hidden_errors = np.dot(
                target_or_weights.transpose(), next_layer_deltas
            )  # Errors in the hidden layer
            self.deltas = hidden_errors * self.activator.activate(
                self.output_array, derivative=True
            )
            # deltas += (next_layers weights * next_layers deltas)

    def update_weights(self, inputs, learningRate):
        change_in_weights = np.dot(self.deltas, inputs.T) * learningRate
        self.weights = np.add(self.weights, change_in_weights)
        self.biases += self.deltas * learningRate

    def display(self):
        print("\tInputs: ", self.inputs)
        print("\tWeights: ", self.weights)
        print("\tBiases: ", self.biases.T)
        print("\tDeltas: ", self.deltas.T)
        print("\toutput_array: ", self.output_array.T)
        print("\tActivation: ", self.activation_function)
