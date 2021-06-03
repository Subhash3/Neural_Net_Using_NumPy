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
network.add_layer(16, activation_function="tanh") 
network.add_layer(16, activation_function="tanh")

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
dataset_handler = Dataset(inputs, outputs)
dataset_handler.make_dataset(input_file, target_file)
data, size = dataset_handler.get_raw_data()
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
def Train(self, dataset: T_Dataset, size, epochs=100, logging=False, epoch_logging=True, prediction_evaulator=None):
	....
	....
```
For Eg: If you want to train your network for 1000 epochs.
```python3
>>> network.Train(data, size, epochs=1000)
```
Notice that I didn't change the value of `logging` as I want the output to be printed for each epoch.


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
>>> network.set_learning_rate(0.1)
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

<br/>
<br/>

## API
#### [`NeuralNetwork`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/NeuralNetwork.md)
#### [`Layer`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/Layer.md)
#### [`Dataset`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/Dataset.md)
#### [`ActivationFunction`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/ActivationFunction.md)
    - Sigmoid
    - Tanh
    - Softmax
    - Identity
    - ReLu
#### [`LossFunctions`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/LossFunctions.md)
    - MSE (Mean Squared Error)
    - CE (Corss Entropy)
#### [`Utils`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/Utils.md)

<br/>
<br/>

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
    - [x] convert all camel-cased vars to snake-case.

    - [ ] API docs
        - [x] Add doc strings to all functions.
        - [x] Make the class/function declarations' docs collapsable.
        - [ ] Merge API md files and embed them in Readme.
        - [ ] Create a section, API, in README to provide documentation for all prototypes.

    - [ ] Implement Batch Training
    - [ ] Write a separate class for Scalers as the scaling methods increase.
    - [ ] Linear and Relu activation functions
    - [ ] Ability to perform regression
    - [ ] Separate out outputlayer from other layers. => Create a separate class for output layer which inherits Layer.


    - [ ] Convolution Nets
    - [ ] Recurrent Nets