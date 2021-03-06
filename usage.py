#!/usr/bin/python3

from src.Model import NeuralNetwork
from src.Dataset import Dataset
import numpy as np

XOR_data = [
    [
        np.array([[0], [0]]),
        np.array([[0]])
    ],
    [
        np.array([[0], [1]]),
        np.array([[1]])
    ],
    [
        np.array([[1], [0]]),
        np.array([[1]])
    ],
    [
        np.array([[1], [1]]),
        np.array([[0]])
    ]
]

size = 4

# input_file = "./datasets/fun/input.csv"
# target_file = "./datasets/fun/target.csv"

# datasetCreator = Dataset(2, 2)
# datasetCreator.makeDataset(input_file, target_file)
# XOR_data, size = datasetCreator.getRawData()

network = NeuralNetwork(2, 2)
network.addLayer(4, activation_function="tanh")
# network.addLayer(16, activation_function="tanh")
network.compile(activation_function="sigmoid")
network.Train(XOR_data, size, epochs=500, logging=False)
# network.epoch_vs_error()

file_to_export = "xor-model.json"
network.export_model(file_to_export)