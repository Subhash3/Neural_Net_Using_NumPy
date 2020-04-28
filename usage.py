#!/usr/bin/python3

from Model import NeuralNetwork
import numpy as np
from Dataset import Dataset

# XOR_data = [
#     [
#         np.array([[0], [0]]),
#         np.array([[0]])
#     ],
#     [
#         np.array([[0], [1]]),
#         np.array([[1]])
#     ],
#     [
#         np.array([[1], [0]]),
#         np.array([[1]])
#     ],
#     [
#         np.array([[1], [1]]),
#         np.array([[0]])
#     ]
# ]

input_file = "./datasets/fun/input.csv"
target_file = "./datasets/fun/target.csv"

datasetCreator = Dataset(2, 2)
datasetCreator.makeDataset(input_file, target_file)
XOR_data, size = datasetCreator.getRawData()

network = NeuralNetwork(2, 2)
network.addLayer(16, activation_function="tanh")
network.addLayer(16, activation_function="tanh")
network.compile(activation_function="sigmoid")
network.Train(XOR_data, size, epochs=1000, logging=False)
network.epoch_vs_error()
network.evaluate()