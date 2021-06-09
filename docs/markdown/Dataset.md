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