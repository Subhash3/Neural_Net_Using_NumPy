from typing import Dict
import random
from math import floor
import numpy as np


def customArgmax(data: Dict[str, float]):
    maxKey = None
    maxValue = None

    for key in data:
        if maxKey == None:
            maxKey = key
        if maxValue == None or maxValue < data[key]:
            maxValue = data[key]
            maxKey = key

    return maxKey


def shuffleArray(array: list):
    arrayCopy = array.copy()
    random.shuffle(arrayCopy)

    return arrayCopy


def splitArr(array: list, ratio: float):
    n = len(array)

    m = floor(n * ratio)

    firstPart: list = array[0: m]
    secondPart: list = array[m: n]

    return [firstPart, secondPart]


def one_hot_encode(num, size) :
    vector = np.array([0]*size)
    vector[num-1] = 1

    return vector.T