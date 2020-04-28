# Feed Forward Neural Networks using NumPy
This library is a modification of my previous one. [Click Here](https://github.com/Subhash3/Neural-Networks/tree/master/Feed_Forward_Networks) to check my previous library.


## Installation  
```bash
$ [sudo] pip3 install neuralnetworks-shine7
``` 

## Development Installation
```bash
$ git clone https://github.com/Subhash3/Neural_Net_Using_NumPy.git
```

## Usage

```python3
>>> from Model import NeuralNetwork
```
### Creating a Neural Network
```python3
inputs = 2
outputs = 1
network = NeuralNetwork(inputs, outputs)

# Add 2 hidden layers with 16 neurons each and activation function 'tanh'
network.addLayer(16, activation_function="tanh") 
network.addLayer(16, activation_function="tanh")

# Finish the neural network by adding the output layer with sigmoid activation function.
network.compile(activation_function="sigmoid")
```
### Building a dataset
The package contains a Dataset class to create a dataset.

```python3
>>> from Dataset import Dataset
```

Make sure you have inputs and target values in seperate files in csv format.

```python3
input_file = "inputs.csv"
target_file = "targets.csv"

# Create a dataset object with the same inputs and outputs defined for the network.
datasetCreator = Dataset(inputs, outputs)
datasetCreator.makeDataset(input_file, target_file)
data, size = datasetCreator.getRawData()
```
### Training The network
The library provides a *Train* function which accepts the dataset, dataset size, and two optional parameters epochs, and logging.
```python3
def Train(dataset, size, epochs=5000, logging=True) :
	....
	....
```
For Eg: If you want to train your network for 1000 epochs.
```python3
>>> network.Train(data, size, epochs=1000)
```
Notice that I didn't change the value of log_outputs as I want the output to printed for each epoch.


### Debugging
Plot a nice epoch vs error graph
```python3
>>> network.epoch_vs_error()
```

Know how well the model performed.
```python3
>>> network.evaluate()
```