import random
from math import floor

import numpy as np

from .Types import T_Output_Array


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

    return vector.reshape(size, 1)


def get_time_required(start, end, epochs):
    seconds = (end - start) * epochs

    minutes = seconds // 60
    seconds = seconds % 60

    hours = minutes // 60
    minutes = minutes % 60
    return f"Estimated Training Time: {hours}hrs::{minutes}min::{round(seconds, 4)}sec"


def judge_prediction(prediction: T_Output_Array, target: T_Output_Array) -> bool:
    is_correct_output: bool = np.argmax(prediction) == np.argmax(target)
    return is_correct_output
