#!/usr/bin/python3

from nicenet import NeuralNetwork
import numpy as np

model_file = "./xor-model.json"
network = NeuralNetwork.load_model(model_file)

input_array = np.array([[0], [0]])
output = network.predict(input_array)
print("Input: ", input_array.T)
print("Output: ", output.T)

network.epoch_vs_error()
network.evaluate()
