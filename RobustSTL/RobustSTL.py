from utils import *
import numpy as np

def NoiseRemoval(sample, H, delta1, delta2):
    '''
    args:
    - delta1: hyperparameters of bilateral filter which for distance
    - delta1: hyperparameters of bilateral filter which for value
    - H: number of neighbors
    '''
    def GetDenoiseValue(idx):
        start_idx, end_index = GetNeighborIdx(len(sample), idx, H)
        idxs= np.arange(start_idx, end_index)
        neighbors = sample[idxs]
        weights = np.array(list(map(lambda j:BilateralFilter(j, idx, sample[j], sample[idx],delta1, delta2), idxs)))
        ## normalization(1/z) = Division np.sum(weights)
        return np.sum(weights * neighbors) / np.sum(weights)
    
    idx_list = np.arange(len(sample))
    denoise_sample = np.array(list(map(GetDenoiseValue, idx_list)))
    return denoise_sample

def RobustSTL(input, seasonal_length, dn1, dn2, H):
    '''
    args:
    - dn1: hyperparameters of bilateral filter in Noise remove, which for distance
    - dn2: hyperparameters of bilateral filter in Noise remove, which for value
    - H: number of neighbors in right and left, which use in NoiseRemoval and SeasonalExtraction
    '''
    sample = input
    iteration = 1
    denoise_sample = NoiseRemoval(sample, H, dn1, dn2)
    #print(len(denoise_sample))
    return "hello world"
