from cvxopt import matrix
from utils import *
import numpy as np
from l1_norm import l1


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
    '''
    - seasonal_length: T in seasonal difference operation
    - lambda1: weight with first order regularization
    - lambda2: weight with second order regularization
    '''
    denoise_sample_length = len(denoise_sample)
    season_diff = denoise_sample[seasonal_length: ] - denoise_sample[ :-seasonal_length]
    assert len(season_diff) == (denoise_sample_length - seasonal_length)
    q = np.concatenate((season_diff, np.zeros([denoise_sample_length*2 - 3])))
    q = np.reshape(q, [len(q), 1])
    q = matrix(q)
    M = GetToeplitz([denoise_sample_length-seasonal_length, denoise_sample_length-1], np.ones([seasonal_length]))
    D = GetToeplitz([denoise_sample_length-2, denoise_sample_length-1], np.array([1,-1]))
    P = np.concatenate([M, lambda1*np.eye(denoise_sample_length-1), lambda2*D], axis=0)
    P = matrix(P)

    delta_trend = l1(P,q)

    relative_trend = GetRaltiveTrend(delta_trend)
    return denoise_sample-relative_trend, relative_trend

def SeasonalityExtraction(detrend_sample, seasonal_length, K, H, ds1, ds2):
    
    '''
    args:
    - detrend_sample: time series data after remove relative trend
    - seasonal_length = T
    - H: number of neighbors in right and left, which use in NoiseRemoval and SeasonalityExtraction
    - K: number of neighborhoods we get from the past, which use in SeasonalityExtraction
    - ds1: hyperparameters of bilateral filter in non-local filtering, which for distance
    - ds2: hyperparameters of bilateral filter in non-local filtering, which for value
    '''
    def GetSeasonalityValue(center):
        idxs = GetSeasonIdx(detrend_sample_length, center, seasonal_length, K, H)
        if idxs.size == 0:
            return detrend_sample[center]

        weight_sample = detrend_sample[idxs]
        weights = np.array(list(map(lambda j: BilateralFilter(j, center, detrend_sample[j], detrend_sample[center],ds1, ds2), idxs)))
        tilda_season = np.sum(weights * weight_sample)/np.sum(weights)
        return tilda_season

    detrend_sample_length = len(detrend_sample)
    idx_list = np.arange(detrend_sample_length)
    tilda_season = np.array(list(map(GetSeasonalityValue, idx_list)))
    
    return tilda_season


def RobustSTL(input, seasonal_length, dn1, dn2, H, lambda1, lambda2, K, ds1, ds2):
    '''
    args:
    - dn1: hyperparameters of bilateral filter in Noise remove, which for distance
    - dn2: hyperparameters of bilateral filter in Noise remove, which for value
    - lambda1: hyperparameters of wieght for first order regularization for trend extraction
    - lambda2: hyperparameters of wieght for second order regularization for trend extraction
    - H: number of neighbors in right and left, which use in NoiseRemoval and SeasonalityExtraction
    - K: number of neighborhoods we get from the past, which use in SeasonalityExtraction
    - ds1: hyperparameters of bilateral filter in non-local filtering, which for distance
    - ds2: hyperparameters of bilateral filter in non-local filtering, which for value
    '''
    sample = input
    iteration = 1
    ## Step1 remove noise in input via bilateral filtering
    denoise_sample = NoiseRemoval(sample, H, dn1, dn2)
    ## Step2 trend extraction with LAD
    detrend_sample , relative_trend = TrendExtraction(denoise_sample, seasonal_length, lambda1, lambda2)
    ## Step3 seaonality extraction with non-local filtering by bilateral filtering
    tilda_season = SeasonalityExtraction(detrend_sample, seasonal_length, K, H, ds1, ds2)
    return "hello world"
