# sigmoid_functions.py

import numpy as np

def sigmoid_basic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))
