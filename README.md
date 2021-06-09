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
<details><summary><code>def __init__(self, I, O, cost="mse"):</code></summary>
<p>

```python
def __init__(self, I, O, cost="mse"):
        """
        Creates a Feed Forward Neural Network.

        Parameters
        ----------
        I : int
            Number of inputs to the network

        O : int
            Number of outputs from the network

        [cost]: string
            The cost/loss function used by the neural network.
            Default value is 'mse' which stands for Mean Squared Error.

            Available options:
                mse => Mean Squared Error
                ce => Cross Entropy

        Returns
        -------
        Doesn't return anything
        """
```
</p>
</details>

<details><summary><code>def set_learning_rate(self, lr):</code></summary>
<p>

```python
def set_learning_rate(self, lr):
        """
        Modifies the learning rate of the network.

        Parameters
        ----------
        lr : float
            New learning rate

        Returns
        -------
        Doesn't return anything
        """
```
</p>
</details>

<details><summary><code>def add_layer(self, num_nodes, activation_function="sigmoid"):</code></summary>
<p>

```python
def add_layer(self, num_nodes, activation_function="sigmoid"):
        """
        Adds a layer to the network.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the hidden layer

        [activation_function] :str
            It is an optional parameter.
            Specifies the activation function of the layer.
            Default value is sigmoid.

            Available options:
                sigmoid,
                tanh,
                linear,
                identity,
                softmax

        Returns
        -------
        Doesn't return anything
        """
```
</p>
</details>

<details><summary><code>def compile(self, activation_function="sigmoid"):</code></summary>
<p>

```python
def compile(self, activation_function="sigmoid"):
        """
        Basically, it just adds the output layer to the network.

        Parameters
        ----------
        [activation_function] :str
            It is an optional parameter.
            Specifies the activation function of the layer.
            Default value is sigmoid.

        Returns
        -------
        Doesn't return anything
        """
```
</p>
</details>

<details><summary><code>def feedforward(self, input_array: T_Feature_Array):</code></summary>
<p>

```python
def feedforward(self, input_array: T_Feature_Array):
        """
        Feeds the given input throughout the network

        Parameters
        ----------
        input_array : T_Feature_Array
            Input to be fed to the network.
            It is columnar vector of size Inputs x 1

        Returns
        -------
        all_outputs : T_Output_Array
            An array of all the outputs produced by each layer.
        """
```
</p>
</details>

<details><summary><code>def backpropagate(self, target: T_Target_Array):</code></summary>
<p>

```python
def backpropagate(self, target: T_Target_Array):
        """
        Backpropagate the error throughout the network
        This function is called inside the model only.

        Parameters
        ----------
        target : np.array()
            It is the ground truth value corresponding to the input.
            It is columnar vector of size Outputs x 1

        Returns
        -------
        Error : float
            # Returns the Mean Squared Error of the particular output
            Returns the error using the specified loss function.
        """
```
</p>
</details>

<details><summary><code>def update_weights(self, input_array: T_Feature_Array):</code></summary>
<p>

```python
def update_weights(self, input_array: T_Feature_Array):
        """
        Update the weights of the network.
        This function is called inside the model only.

        Parameters
        ----------
        input_array : np.array()
            It is the input fed to the network
            It is columnar vector of size Inputs x 1

        Returns
        -------
        Doesn't return anything
        """
```
</p>
</details>

<details><summary><code>def Train(self, dataset: T_Dataset, size, epochs=100, logging=False, epoch_logging=True, prediction_evaulator=None):</code></summary>
<p>

```python
def Train(self, dataset: T_Dataset, size, epochs=100, logging=False, epoch_logging=True, prediction_evaulator=None):
        """
        Trains the neural network using the given dataset.

        Parameters
        ----------
        dataset : T_Dataset

        size : int
            Size of the dataset

        [epochs] : int
            An optional parameter.
            Number of epochs to train the network. Default value is 5000

        [logging] : bool
            An optional parameter.
            If its true, all outputs from the network will be logged out onto STDOUT for each epoch.

        [epoch_logging] : bool
            An optional parameter.
            If it is true, Error in each epoch will be logged to STDOUT.

        [prediction_evaulator]: (prediction: T_Output_Array, target: T_Output_Array) -> bool
            An optional parameter.
            Used to evaluate the fed forward output with the actual target.
            Default value is 'Utils.judge_prediction' function.

        Returns
        -------
        Doesn't return anything.
        """
```
</p>
</details>

<details><summary><code>def predict(self, input_array: T_Feature_Array):</code></summary>
<p>

```python
def predict(self, input_array: T_Feature_Array):
        """
        Predicts the output using a given input.

        Parameters
        ----------
        input_array : np.array()
            It is columnar vector of size Inputs x 1
            It is the input fed to the network

        Returns
        -------
        prediction : np.array()
            Predicted value produced by the network.
        """
```
</p>
</details>

<details><summary><code>def epoch_vs_error(self):</code></summary>
<p>

```python
def epoch_vs_error(self):
        """
        Plot error vs epoch graph

        Parameters
        ----------
        Doesn't accept any parameters

        Returns
        -------
        Doesn't return anything
        """
```
</p>
</details>

<details><summary><code>def evaluate(self):</code></summary>
<p>

```python
def evaluate(self):
        """
        Print the basic information about the network.
        Like accuracy, error ..etc.

        Parameters
        ----------
        Doesn't accept any parameters

        Returns
        -------
        Doesn't return anything
        """
```
</p>
</details>

<details><summary><code>def display(self):</code></summary>
<p>

```python
def display(self):
        """
        Print the information of each layer of the network.
        It can be used to debug the network!

        Parameters
        ----------
        Doesn't accept any parameters

        Returns
        -------
        Doesn't return anything
        """
```
</p>
</details>

<details><summary><code>def export_model(self, filename):</code></summary>
<p>

```python
def export_model(self, filename):
        """
        Export the model to a json file

        Parameters
        ----------
        filename: str
            File name to export model

        Returns
        -------
        Doesn't return anything
        """
```
</p>
</details>

<details><summary><code>def load_model(filename):</code></summary>
<p>

```python
def load_model(filename):
        """
        Load model from an eported (json) model

        Parameters
        ----------
        filename : str
            Exported model (json) file

        Returns
        -------
        brain : NeuralNetwork
            NeuralNetwork object
        """
```
</p>
</details>

<br/>

#### [`Layer`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/Layer.md)
<details><summary><code>def __init__(self, num_nodes, inputs, activation_function, loss_function):</code></summary>
<p>

```python
def __init__(self, num_nodes, inputs, activation_function, loss_function):
        """
            Layer constructor

            Parameters
            ----------
            num_nodes : int
                No. of nodes in the layer

            inputs : int
                No. of inputs to the layer

            activation_function

            Returns
            -------
            None
        """
```
</p>
</details>

<details><summary><code>def feed(self, input_array: T_Feature_Array) -> T_Output_Array:</code></summary>
<p>

```python
def feed(self, input_array: T_Feature_Array) -> T_Output_Array:
        """
            Feeds the given input array to a particular layer.

            Parameters
            ----------
            input_array: T_Feature_Array
                Input array to be fed to the layer

            Returns
            -------
            output_array: T_Output_Array
        """
```
</p>
</details>

<details><summary><code>def activate(self, x):</code></summary>
<p>

```python
def activate(self, x):
        """
            Passes the output array to an activation function.

            Parameters
            ----------
            x
                Output array from a layer

            Returns
            -------
            Activated output
        """
```
</p>
</details>

<details><summary><code>def calculate_gradients(self, target_or_weights, layer_type, next_layer_deltas=None):</code></summary>
<p>

```python
def calculate_gradients(self, target_or_weights, layer_type, next_layer_deltas=None):
        """
            Calculates the gradients for each weight and bias

            Parameters
            ----------
            target_or_weights
                This is either targers array of weights matrix.
                Specifically, it'll be the targets array while computing the gradients for the output layer
                and weights matrix of the next layer.

            layer_type
                This will either be "hidden" or "output"

            [next_layer_deltas]
                This is (not exactly) an optional parameter.
                This will be passed only while computing the gradients of a hidden layer.

            Returns
            -------
                Doesn't return anything. But stores the gradients as a class attribute.
        """
```
</p>
</details>

<details><summary><code>def update_weights(self, inputs, learning_rate):</code></summary>
<p>

```python
def update_weights(self, inputs, learning_rate):
        """
            Tweak the weights of the layer.

            Parameters
            ----------
            inputs: T_Feature_Array
                Input to this network

            learning_rate: float
                Learning rate of the entire network.

            Returns
            -------
            Doesn't return anything.
        """
```
</p>
</details>

<details><summary><code>def display(self):</code></summary>
<p>

```python
def display(self):
        """
            Display the metadata of the layer.
        """
```
</p>
</details>

<br/>

#### [`Dataset`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/Dataset.md)
<details><summary><code>def __init__(self, I, O):</code></summary>
<p>

```python
def __init__(self, I, O):
        """
            Dataset Constructor

            Parameters
            ----------
            I: int
                No of inputs
            O: int
                No of outputs

        """
```
</p>
</details>

<details><summary><code>def make_dataset(self, input_file, target_file):</code></summary>
<p>

```python
def make_dataset(self, input_file, target_file):
        """
            Creates a dataset

            Parameters
            ----------
            input_file: str
                csv file containing the features/inputs.
            target_file: str
                csv file containing the targets.

            Returns
            -------
            Doesn't return anything.
        """
```
</p>
</details>

<details><summary><code>def get_raw_data(self):</code></summary>
<p>

```python
def get_raw_data(self):
        """
            Returns the dataset which was made earlier in make_dataset method

            Parameters
            ----------
            Doesn't accept anything

            Returns
            Tuple[T_Dataset, int]
                Dataset and its size
        """
```
</p>
</details>

<details><summary><code>def display(self):</code></summary>
<p>

```python
def display(self):
        """
            Prints the dataset
        """
```
</p>
</details>

<details><summary><code>def open_file(filename):</code></summary>
<p>

```python
def open_file(filename):
        """
            Just a helper function to open a given file and handle errors if any.

            Parameters
            ----------
            filename: str
                Filename to be opened

            Returns
            -------
            fhand
                A filehandler corresponding to the given file.
        """
```
</p>
</details>

<details><summary><code>def get_min_max_values(self, array: T_Dataset):</code></summary>
<p>

```python
def get_min_max_values(self, array: T_Dataset):
        """
            Computes the min and max of each feature, and each target label of the entire dataset.

            Parameters
            ----------
            array : List[List[np.array]]
                List of datasamples
                datasample = [
                    column vector of features,
                    column vector of of targets
                ]

            Returns
            -------
            Tuple[List[float]]
                min and max values of features and targets
                (min_max_of_features, min_max_of_targets)
                min_max_of_features = List[[min_of_ith_feature, max_of_ith_feature]]
                min_max_of_targets = List[[min_of_ith_target, max_of_ith_target]]
        """
```
</p>
</details>

<details><summary><code>def scale_data(self, array: T_Dataset, size):</code></summary>
<p>

```python
def scale_data(self, array: T_Dataset, size):
        """
            Scales the data using min-max scaling method.

            parameters
            ----------
            array: T_Dataset
                Dataset to be scaled.

            size: int
                Size of the given dataset.

            Returns
            -------
            array: T_Dataset
                Scaled dataset.
        """
```
</p>
</details>

<br/>

#### [`ActivationFunction`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/ActivationFunction.md)


<br/>

#### [`LossFunctions`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/LossFunctions.md)


<br/>

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

    - [x] API docs
        - [x] Add doc strings to all functions.
        - [x] Make the class/function declarations' docs collapsable.
        - [x] Merge API md files and embed them in Readme.
        - [x] Create a section, API, in README to provide documentation for all prototypes.

    - [ ] Implement Batch Training
    - [ ] Write a separate class for Scalers as the scaling methods increase.
    - [ ] Linear and Relu activation functions
    - [ ] Ability to perform regression
    - [ ] Separate out outputlayer from other layers. => Create a separate class for output layer which inherits Layer.


    - [ ] Convolution Nets
    - [ ] Recurrent Nets