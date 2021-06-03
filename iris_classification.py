#!/usr/bin/python3

from nicenet import NeuralNetwork
from nicenet import Dataset
from helpers import shuffle_array, split_arr
import numpy as np

inputs = 4
outputs = 3
network = NeuralNetwork(inputs, outputs, cost="ce")
network.add_layer(8, activation_function="sigmoid")
network.add_layer(8, activation_function="sigmoid")
network.compile(activation_function="softmax")
network.set_learning_rate(0.1)

dataset_handler = Dataset(inputs, outputs)
dataset_handler.make_dataset(
    './datasets/Iris/inputs.csv', './datasets/Iris/targets.csv')
data, size = dataset_handler.get_raw_data()
# data = dataset_handler.scale_data(data, size)
data = shuffle_array(data)
training, testing = split_arr(data, 3/4)
# print(len(training))

network.Train(training, len(training), epochs=50,
              logging=False, epoch_logging=False)
network.evaluate()
network.epoch_vs_error()

network.export_model('iris_model.json')

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
print("Testing accuracy:", correct*100/total)

# new_network = NeuralNetwork.load_model('iris_model.json')
# correct = 0
# total = 0
# for sample in testing :
#     features = sample[0]
#     prediction = new_network.predict(features)
#     actual = sample[1]
#     p = np.argmax(prediction)
#     a = np.argmax(actual.T[0])

#     if p == a:
#         correct += 1
#     total += 1
# print("Testing accuracy:", correct*100/total)
