from scipy.linalg import toeplitz
import numpy as np
import math


def BilateralFilter(j, t, y_j, y_t, delta1, delta2):
    pow1 = -1.0 * (math.fabs(j-t)**2.0) / (2.0 * (delta1**2))
    pow2 = -1.0 * (math.fabs(y_j-y_t)**2.0) / (2.0 * (delta2**2))
    weight = math.exp(pow1) * math.exp(pow2)
    return weight


def GetNeighborIdx(sample_length, Center, H):
    '''
    Let i = Center
    Then, return i-H, i-(H-1), ..., i, i+(H-1), i+H+1

    Due to head and tail may be less than H elements,
    using max() and min() to select head and tail element's index
    '''
    return[np.max([0, Center-H]), np.min([sample_length, Center+H+1])]

def GetToeplitx(shape_size, entry):
    height, width = shape_size
    entry_length = len(entry)
    assert np.ndim(entry) < 2
    if entry_length < 1:
        return np.zeros(shape_size)
    row = np.concatenate([entry[:1], np.zeros(height - 1)])
    col = np.concatenate([np.array(entry), np.zeros(width - entry_length)])

    return toeplitz(row, col)