# Feed Forward Neural Networks using NumPy
This library is a modification of my previous one. [Click Here](https://github.com/Subhash3/Neural-Networks/tree/master/Feed_Forward_Networks) to check my previous library.


## Installation  
```bash
$ [sudo] pip3 install nicenet
``` 

## Development Installation
```bash
$ git clone https://github.com/Subhash3/Neural_Net_Using_NumPy.git
```

## Usage

```python3
>>> from nicenet import NeuralNetwork
```
### Creating a Neural Network
```python3
inputs = 2
outputs = 1
network = NeuralNetwork(inputs, outputs, cost="mse")

# Add 2 hidden layers with 16 neurons each and activation function 'tanh'
network.addLayer(16, activation_function="tanh") 
network.addLayer(16, activation_function="tanh")

# Finish the neural network by adding the output layer with sigmoid activation function.
network.compile(activation_function="sigmoid")
```
### Building a dataset
The package contains a Dataset class to create a dataset.

```python3
>>> from nicenet import Dataset
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

If you want to manually make a dataset, follow these rules:
- Dataset must be a list of data samples.
- A data sample is a tuple containing inputs and target values.
- Input and target values are column vector of size (inputs x 1) and (outputs x 1) respectively.

For eg, a typical XOR data set looks something like :
```python3
>>> XOR_data = [
    (
        np.array([[0], [0]]),
        np.array([[0]])
    ),
    (
        np.array([[0], [1]]),
        np.array([[1]])
    ),
    (
        np.array([[1], [0]]),
        np.array([[1]])
    ),
    (
        np.array([[1], [1]]),
        np.array([[0]])
    )
]
>>> size = 4
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

To take a look at all the layers' info
```python3
>>> network.display()
```

Sometimes, learning rate might have to be altered for better convergence.
```python3
>>> network.setLearningRate(0.1)
```

### Exporting Model
You can export a trained model to a json file which can be loaded and used for predictions in the future.
```python3
filename = "model.json"
network.export_model(filename)
```

### Load Model
To load a model from an exported model (json) file.
load\_model is a static function, so you must not call this on a NeuralNetwork object!.
```python3
filename = "model.json"
network = NeuralNetwork.load_model(filename)
```

### Todo
    - [x] Generalize the gradient descent algorithm
        - [x] Generalise the loss function => Write a separate class for it!
    - [x] Implement Cross Entropy Loss
    - [ ] Data scaling
        - [x] Min Max scaler
        - [ ] Data Standardization
    - [x] Change the datasample type to a tuple instead of a list.
    - [x] Show Progress bar if epoch_logging is False

    - [x] Use a function as a parameter to Train method to compare predictions and actual targets.
    - [x] Add doc strings to all functions.
    - [ ] Implement Batch Training
    - [ ] Write a separate class for Scalers as the scaling methods increase.
    - [ ] Linear and Relu activation functions
    - [ ] Ability to perform regression
    - [ ] Separate out outputlayer from other layers. => Create a separate class for output layer which inherits Layer.


    - [ ] Convolution Nets
    - [ ] Recurrent Nets