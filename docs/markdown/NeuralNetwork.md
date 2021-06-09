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