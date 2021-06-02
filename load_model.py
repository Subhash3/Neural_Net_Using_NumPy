#!/usr/bin/python3

import numpy as np
from src.Model import NeuralNetwork

model_file = "./xor-model.json"
network = NeuralNetwork.load_model(model_file)

input_array = np.array([[0], [0]])
output = network.predict(input_array)
print("Input: ", input_array.T)
print("Output: ", output.T)

network.epoch_vs_error()
network.evaluate()
