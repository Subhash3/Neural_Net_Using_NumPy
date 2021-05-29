import random
from math import floor
import numpy as np


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


def one_hot_encode(num, size):
    vector = np.array([0]*size)
    vector[num-1] = 1

    return vector.reshape(size, 1)


def get_time_required(start, end, epochs):
    seconds = (end - start) * epochs

    minutes = seconds // 60
    seconds = seconds % 60

    hours = minutes // 60
    minutes = minutes % 60
    return f"Estimated Training Time: {hours}hrs::{minutes}min::{round(seconds, 4)}sec"
