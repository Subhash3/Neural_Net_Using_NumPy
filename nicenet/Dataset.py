import numpy as np
from typing import List
from .Types import T_Feature_Array, T_Target_Array, T_Data_Sample, T_Dataset


# Dataset class
class Dataset():
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
        self.I = I
        self.O = O
        self.size = 0
        self.dataset: T_Dataset = list()

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

        input_handler = self.open_file(inputFile)
        target_handler = self.open_file(targetFile)

        if not input_handler or not target_handler:
            print("Unable to create Dataset")
            return
        input_lines = input_handler.readlines()
        target_lines = target_handler.readlines()

        for inp, tar in zip(input_lines, target_lines):
            input_array = list(map(float, inp.split(',')))
            target_array = list(map(float, tar.split(',')))

            features: T_Feature_Array = np.reshape(input_array, (self.I, 1))
            targets: T_Target_Array = np.reshape(target_array, (self.O, 1))
            sample: T_Data_Sample = (features, targets)
            self.dataset.append(sample)
            self.size += 1

    def modify_lists(self, input_array, target_array):
        sample = list()
        sample.append(np.reshape(input_array, (self.I, 1)))
        sample.append(np.reshape(target_array, (self.O, 1)))
        self.dataset.append(sample)
        self.size += 1

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
        return self.dataset, self.size

    def display(self):
        """
            Prints the dataset
        """
        for i in range(self.size):
            sample = self.dataset[i]
            print("Data Sample:", i+1)
            print("\tInput: ", sample[0])
            print("\tTarget: ", sample[1])

    @staticmethod
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
        try:
            fhand = open(filename)
        except Exception as e:
            print("[!!] Exception Occurred while Opening", filename, e)
            return None
        return fhand

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

        no_of_features = array[0][0].shape[0]
        no_of_targets = array[0][1].shape[0]

        min_max_of_features = list()
        # print(min_max_of_features)
        for i in range(no_of_features):
            ith_feature_values = [sample[0][i][0] for sample in array]
            # print(ith_feature_values)
            min_of_ith_feature = min(ith_feature_values)
            max_of_ith_feature = max(ith_feature_values)
            min_max_of_features.append(
                [min_of_ith_feature, max_of_ith_feature])

        # print(min_max_of_features)

        min_max_of_targets = list()
        # print(min_max_of_targets)
        for i in range(no_of_targets):
            ith_target_values = [sample[1][i][0] for sample in array]
            # print(ith_target_values)
            min_of_ith_target = min(ith_target_values)
            max_of_ith_target = max(ith_target_values)
            min_max_of_targets.append([min_of_ith_target, max_of_ith_target])

        # print(min_max_of_targets)

        return min_max_of_features, min_max_of_targets

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

        if size == 0:
            return

        # print(array)

        no_of_features = array[0][0].shape[0]
        no_of_targets = array[0][1].shape[0]

        min_max_of_features, min_max_of_targets = self.get_min_max_values(
            array)

        for i in range(size):
            sample = array[i]
            features = sample[0]
            targets = sample[1]

            for j in range(no_of_features):
                min_of_jth_feature = min_max_of_features[j][0]
                max_of_jth_feature = min_max_of_features[j][1]
                features[j][0] = (features[j][0] - min_of_jth_feature) / \
                    (max_of_jth_feature - min_of_jth_feature)

            for j in range(no_of_targets):
                min_of_jth_target = min_max_of_targets[j][0]
                max_of_jth_target = min_max_of_targets[j][1]
                targets[j][0] = (targets[j][0] - min_of_jth_target) / \
                    (max_of_jth_target - min_of_jth_target)

        return array
