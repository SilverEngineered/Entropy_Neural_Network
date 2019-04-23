import numpy as np
from scipy.stats import entropy
from math import log, e
import pandas as pd

import timeit

def singleEntropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

def averageEntropy(set):
    sum = 0
    for item in set:
        sum += singleEntropy(item)
    return sum / len(set)