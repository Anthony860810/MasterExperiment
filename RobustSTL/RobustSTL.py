from numpy.matrixlib.defmatrix import matrix
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

def TrendExtraction(denoise_sample, seasonal_length, lambda1, lambda2):
    denoise_sample_length = len(denoise_sample)
    season_diff = denoise_sample[seasonal_length: ] - denoise_sample[ :-seasonal_length]
    assert len(season_diff) == (denoise_sample_length - seasonal_length)
    q = np.concatenate((season_diff, np.zeros([denoise_sample_length*2 - 3])))
    q = np.reshape(q, [len(q), 1])
    q = matrix(q)
    M = GetToeplitx([denoise_sample_length-seasonal_length, denoise_sample_length-1], np.ones([seasonal_length]))
    D = GetToeplitx([denoise_sample_length-2, denoise_sample_length-1], np.array([1,-1]))
    P = np.concatenate([M, lambda1*np.eye(denoise_sample_length-1), lambda2*D], axis=0)
    P = matrix(P)
    
    return 0


def RobustSTL(input, seasonal_length, dn1, dn2, H, lambda1, lambda2):
    '''
    args:
    - dn1: hyperparameters of bilateral filter in Noise remove, which for distance
    - dn2: hyperparameters of bilateral filter in Noise remove, which for value
    - lambda1: hyperparameters of wieght for first order regularization for trend extraction
    - lambda2: hyperparameters of wieght for second order regularization for trend extraction
    - H: number of neighbors in right and left, which use in NoiseRemoval and SeasonalExtraction

    '''
    sample = input
    iteration = 1
    ## Step1 remove noise in input via bilateral filtering
    denoise_sample = NoiseRemoval(sample, H, dn1, dn2)
    detrend_sample = TrendExtraction(denoise_sample, seasonal_length, lambda1, lambda2)
    #print(len(denoise_sample))
    return "hello world"
