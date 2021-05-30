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


```python
def display(self):
        """
            Prints the dataset
        """
```


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
