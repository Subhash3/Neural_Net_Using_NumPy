import random
from math import floor
from typing import Dict

import numpy as np


def custom_argmax(data: Dict[str, float]):
    max_key = None
    max_value = None

    for key in data:
        if not max_key:
            max_key = key
        if not max_value or max_value < data[key]:
            max_value = data[key]
            max_key = key

    return max_key


def shuffle_array(array: list):
    arrayCopy = array.copy()
    random.shuffle(arrayCopy)

    return arrayCopy


def split_arr(array: list, ratio: float):
    n = len(array)

    m = floor(n * ratio)

    firstPart: list = array[0:m]
    secondPart: list = array[m:n]

    return [firstPart, secondPart]


def one_hot_encode(num, size):
    vector = np.array([0] * size)
    vector[num - 1] = 1

    return vector.T
