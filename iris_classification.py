#!/usr/bin/python3

import numpy as np

from helpers import shuffle_array, split_arr
from nicenet import Dataset, NeuralNetwork

inputs = 4
outputs = 3
network = NeuralNetwork(inputs, outputs, cost="ce")
network.add_layer(8, activation_function="sigmoid")
network.add_layer(8, activation_function="sigmoid")
network.compile(activation_function="softmax")
network.set_learning_rate(0.1)

datasetHandler = Dataset(inputs, outputs)
datasetHandler.makeDataset("./datasets/Iris/inputs.csv", "./datasets/Iris/targets.csv")
data, size = datasetHandler.getRawData()
# data = datasetHandler.scaleData(data, size)
data = shuffle_array(data)
training, testing = split_arr(data, 3 / 4)
# print(len(training))

network.fit(training, len(training), epochs=50, logging=False, epoch_logging=False)
network.evaluate()
network.epoch_vs_error()

network.export_model("iris_model.json")

correct = 0
total = 0
for sample in testing:
    features = sample[0]
    prediction = network.predict(features)
    actual = sample[1]
    p = np.argmax(prediction)
    a = np.argmax(actual.T[0])

    if p == a:
        correct += 1
    total += 1
print("Testing accuracy:", correct * 100 / total)

# newNetwork = NeuralNetwork.load_model('iris_model.json')
# correct = 0
# total = 0
# for sample in testing :
#     features = sample[0]
#     prediction = newNetwork.predict(features)
#     actual = sample[1]
#     p = np.argmax(prediction)
#     a = np.argmax(actual.T[0])

#     if p == a:
#         correct += 1
#     total += 1
# print("Testing accuracy:", correct*100/total)
