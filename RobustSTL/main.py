from RobustSTL import RobustSTL
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt

Data = "1c4753a1-3512-4b65-a3f8-8371dea8edd4"
DataPath = "/home/tony/python_project/anomaly_detection_data/data/"+Data+".csv"


def main(DataPath):
    df = pd.read_csv(DataPath)
    data = df['value'].to_numpy()
    result = RobustSTL(data, seasonal_length=5, dn1=1.0, dn2=1.0, H=5, lambda1=10, lambda2=0.5, K=2, ds1=1.0, ds2=1.0)
    #print(result)


if __name__ == '__main__':
    main(DataPath)
