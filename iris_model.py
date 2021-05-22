#!/usr/bin/python3
from src import NeuralNetwork
from src import Dataset
from helpers import shuffleArray, splitArr

network = NeuralNetwork(4, 1)

network.addLayer(8, activation_function="tanh")
network.addLayer(8, activation_function="tanh")
network.compile(activation_function="sigmoid")

datasetCreator = Dataset(4, 1)
datasetCreator.makeDataset(
    './datasets/Iris/inputs.csv', './datasets/Iris/targets.csv')
data, size = datasetCreator.getRawData()

data = shuffleArray(data)
training, testing = splitArr(data, 3/4)

network.Train(training, len(training), 100, False)
# network.epoch_vs_error()
network.evaluate()
network.export_model('./iris_model.json')

correct = 0
total = 0
for sample in testing:
    prediction = network.predict(sample[0])
    if round(prediction[0][0], 1) == sample[1][0][0]:
        correct += 1
    total += 1
print(correct/total, correct, total)
