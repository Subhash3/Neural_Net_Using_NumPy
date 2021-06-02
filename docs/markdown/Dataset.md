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

<details><summary><code>def makeDataset(self, inputFile, targetFile):</code></summary>
<p>

```python
def makeDataset(self, inputFile, targetFile):
        """
            Creates a dataset

            Parameters
            ----------
            inputFile: str
                csv file containing the features/inputs.
            targetFile: str
                csv file containing the targets.

            Returns
            -------
            Doesn't return anything.
        """
```
</p>
</details>

<details><summary><code>def getRawData(self):</code></summary>
<p>

```python
def getRawData(self):
        """
            Returns the dataset which was made earlier in makeDataset method

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

<details><summary><code>def openFile(filename):</code></summary>
<p>

```python
def openFile(filename):
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

<details><summary><code>def scaleData(self, array: T_Dataset, size):</code></summary>
<p>

```python
def scaleData(self, array: T_Dataset, size):
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