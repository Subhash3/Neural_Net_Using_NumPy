import numpy as np
from typing import List


# Dataset class
class Dataset():
    def __init__(self, I, O, scale=False):
        self.I = I
        self.O = O
        self.size = 0
        self.dataset = list()
        self.scale = scale

    def makeDataset(self, inputFile, targetFile):
        input_handler = self.openFile(inputFile)
        target_handler = self.openFile(targetFile)

        if not input_handler or not target_handler:
            print("Unable to create Dataset")
            return
        input_lines = input_handler.readlines()
        target_lines = target_handler.readlines()

        for inp, tar in zip(input_lines, target_lines):
            input_array = list(map(float, inp.split(',')))
            target_array = list(map(float, tar.split(',')))

            sample = list()
            if self.scale:
                input_array = self.scaleData(input_array, self.I)
                target_array = self.scaleData(target_array, self.O)

            sample.append(np.reshape(input_array, (self.I, 1)))
            sample.append(np.reshape(target_array, (self.O, 1)))
            self.dataset.append(sample)
            self.size += 1

    def modifyLists(self, input_array, target_array):
        sample = list()
        sample.append(np.reshape(input_array, (self.I, 1)))
        sample.append(np.reshape(target_array, (self.O, 1)))
        self.dataset.append(sample)
        self.size += 1

    def getRawData(self):
        return self.dataset, self.size

    def display(self):
        for i in range(self.size):
            sample = self.dataset[i]
            print("Data Sample:", i+1)
            print("\tInput: ", sample[0])
            print("\tTarget: ", sample[1])

    @staticmethod
    def openFile(filename):
        try:
            fhand = open(filename)
        except Exception as e:
            print("[!!] Exception Occurred while Opening", filename, e)
            return None
        return fhand

    def get_min_max_values(self, array: List[List[np.array]]):
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

    def scaleData(self, array: List[List[np.array]], size, scale_range=(0, 1)):
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
