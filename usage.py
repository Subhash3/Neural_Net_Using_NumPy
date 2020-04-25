#!/usr/bin/python3

from Model import NeuralNetwork
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

network = NeuralNetwork(2, 1)
network.addLayer(2, activation_function="sigmoid")
# network.addLayer(2, activation_function="sigmoid")
network.compile(activation_function="sigmoid")
network.Train(XOR_data, 4, epochs=3000, logging=True)