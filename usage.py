#!/usr/bin/python3

from Model import NeuralNetwork
import numpy as np

Dataset = [
    [
        np.array([[0, 0]]),
        np.array([[0]])
    ],
    [
        np.array([[0, 1]]),
        np.array([[1]])
    ],
    [
        np.array([[1, 0]]),
        np.array([[1]])
    ],
    [
        np.array([[1, 1]]),
        np.array([[1]])
    ]
]

network = NeuralNetwork(2, 1)
network.compile(activation_function="sigmoid")
network.Train(Dataset, 4, epochs=20)