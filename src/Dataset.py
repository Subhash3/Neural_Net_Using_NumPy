#!/usr/bin/python3

import numpy as np

class DataSample() :
    def __init__(self) :
        self.input_array = None
        self.targets = None

class Dataset() :
    def __init__(self, I, O, split=True) :
        self.I = I
        self.O = O
        self.size = 0
        self.dataset =  list()

    def makeDataset(self, inputFile, targetFile) :
        input_handler = self.openFile(inputFile)
        target_handler = self.openFile(targetFile)

        if not input_handler or not target_handler :
            print("Unable to create Dataset")
            return
        input_lines = input_handler.readlines()
        target_lines = target_handler.readlines()

        for inp, tar in zip(input_lines, target_lines) :
            input_array = list(map(float, inp.split(',')))
            target_array = list(map(float, tar.split(',')))

            sample = DataSample()
            sample.input_array = np.reshape(input_array, (self.I, 1))
            sample.targets = np.reshape(target_array, (self.O, 1))
            self.dataset.append(sample)
            self.size += 1
    
    def modifyLists(self, input_array, target_array) :
        sample = DataSample()
        sample.input_array = np.reshape(input_array, (self.I, 1))
        sample.targets = np.reshape(target_array, (self.O, 1))
        self.dataset.append(sample)
        self.size += 1

    def getRawData(self) :
        return self.dataset, self.size
    
    def display(self) :
        for i in range(self.size) :
            sample = self.dataset[i]
            print("Data Sample:", i+1)
            print("\tInput: ", sample.input_array)
            print("\tTarget: ", sample.targets)
    
    @staticmethod
    def openFile(filename) :
        try :
            fhand = open(filename)
        except Exception as e:
            print("[!!] Exception Occurred while Opening", filename, e)
            return None
        return fhand