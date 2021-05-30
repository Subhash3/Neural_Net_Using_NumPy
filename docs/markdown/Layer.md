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


```python
def update_weights(self, inputs, learningRate):
        """
            Tweak the weights of the layer.

            Parameters
            ----------
            inputs: T_Feature_Array
                Input to this network

            learningRate: float
                Learning rate of the entire network.

            Returns
            -------
            Doesn't return anything.
        """
```


```python
def display(self):
        """
            Display the metadata of the layer.
        """
```
