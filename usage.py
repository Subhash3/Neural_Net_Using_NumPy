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

input_file = "./datasets/XOR/input.csv"
target_file = "./datasets/XOR/target.csv"

datasetCreator = Dataset(2, 1)
datasetCreator.makeDataset(input_file, target_file)
XOR_data, size = datasetCreator.getRawData()

network = NeuralNetwork(2, 1)
network.addLayer(2, activation_function="sigmoid")
network.addLayer(2, activation_function="sigmoid")
network.compile(activation_function="sigmoid")
network.Train(XOR_data, size, epochs=3000, logging=False)
network.epoch_vs_error()
network.evaluate()