#!/usr/bin/python3

from nicenet import NeuralNetwork
from nicenet import Dataset
import numpy as np

XOR_data = [
    (np.array([[0], [0]]), np.array([[0]])),
    (np.array([[0], [1]]), np.array([[1]])),
    (np.array([[1], [0]]), np.array([[1]])),
    (np.array([[1], [1]]), np.array([[0]])),
]

size = 4

# input_file = "./datasets/fun/input.csv"
# target_file = "./datasets/fun/target.csv"

# datasetCreator = Dataset(2, 2)
# datasetCreator.makeDataset(input_file, target_file)
# XOR_data, size = datasetCreator.getRawData()

network = NeuralNetwork(2, 1, cost="ce")
network.setLearningRate(0.1)
network.addLayer(8, activation_function="tanh")
network.addLayer(8, activation_function="tanh")
network.addLayer(8, activation_function="tanh")
network.addLayer(8, activation_function="tanh")
network.compile(activation_function="sigmoid")
network.Train(XOR_data, size, epochs=5, logging=False)
# network.epoch_vs_error()
network.evaluate()

file_to_export = "xor-model.json"
network.export_model(file_to_export)

test = np.reshape([1, 1], (2, 1))
out = network.predict(test)
print(out)

new_network = NeuralNetwork.load_model(file_to_export)
test = np.reshape([1, 1], (2, 1))
out = network.predict(test)
print(out)
