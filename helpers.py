from typing import Dict
import random
from math import floor
import numpy as np


def customArgmax(data: Dict[str, float]):
    max_key = None
    max_value = None

    for key in data:
        if max_key == None:
            max_key = key
        if max_value == None or max_value < data[key]:
            max_value = data[key]
            max_key = key

    return max_key


def shuffle_array(array: list):
    array_copy = array.copy()
    random.shuffle(array_copy)

    return array_copy


def split_arr(array: list, ratio: float):
    n = len(array)

    m = floor(n * ratio)

    first_part: list = array[0: m]
    second_part: list = array[m: n]

    return [first_part, second_part]


def one_hot_encode(num, size):
    vector = np.array([0]*size)
    vector[num-1] = 1

    return vector.T
