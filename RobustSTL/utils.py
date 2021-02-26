from scipy.linalg import toeplitz
import numpy as np
import math


def BilateralFilter(j, t, y_j, y_t, delta1, delta2):
    pow1 = -1.0 * (math.fabs(j-t)**2.0) / (2.0 * (delta1**2))
    pow2 = -1.0 * (math.fabs(y_j-y_t)**2.0) / (2.0 * (delta2**2))
    weight = math.exp(pow1) * math.exp(pow2)
    return weight


def GetNeighborIdx(sample_length, center, H):
    '''
    Let i = center
    Then, return i-H, i-(H-1), ..., i, i+(H-1), i+H+1

    Due to head and tail may be less than H elements,
    using max() and min() to select head and tail element's index
    '''
    return[np.max([0, center-H]), np.min([sample_length, center+H+1])]

def GetToeplitz(shape_size, entry):
    '''
    Generate toeplitz matrix
    '''
    height, width = shape_size
    entry_length = len(entry)
    assert np.ndim(entry) < 2
    if entry_length < 1:
        return np.zeros(shape_size)
    row = np.concatenate([entry[:1], np.zeros(height - 1)])
    col = np.concatenate([np.array(entry), np.zeros(width - entry_length)])

    return toeplitz(row, col)

def GetRaltiveTrend(delta_trend):
    init_trend = np.array([0])
    idxs = np.arange(len(delta_trend))
    relative_trend = np.array(list(map(lambda idx: np.sum(delta_trend[:idx]), idxs)))
    relative_trend = np.concatenate((init_trend, relative_trend))
    return relative_trend

def GetNeighborhoodRange(sample_length, center, H):
    start_idx, end_idx = GetNeighborIdx(sample_length, center, H)
    return np.arange(start_idx, end_idx)

def GetSeasonIdx(sample_length, center, T, K, H):
    ## Check how many neighborhoods we can use
    num_neighborhood = np.min([K, int(center/T)])+1
    
    ## Get nighborhood centers
    if center < T:
        neighborhoods_center = center + np.arange(0, num_neighborhood)*(-1*T)
    else:
        neighborhoods_center = center + np.arange(1, num_neighborhood)*(-1*T)
    
    ## Each neighborhoods idxs
    idxs = list(map(lambda idx: GetNeighborhoodRange(sample_length, idx, H), neighborhoods_center))
    season_idxs = []
    for item in idxs:
        season_idxs += list(item)
    season_idxs = np.array(season_idxs)
    return season_idxs

