import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import os
import matplotlib.pyplot as plt

DataPath = "../YahooBenchmark/A4Benchmark/"
OutputPath = "../../anomaly_detection_data/YahooBenchmark/A4Benchmark_stl/"


for fileName in os.listdir(DataPath):
    if(fileName[0:14]=="A4Benchmark-TS"):
        file = os.path.join(DataPath,fileName)
        print(file)
        df = pd.read_csv(file)
        y = df['value'].to_numpy()
        res = STL(y, period=30, robust=True).fit()
        '''plt.plot(df["value"], label="value")
        plt.plot(res.trend, label="trend")
        plt.savefig(OutputPath+fileName[:-4]+".png")
        plt.clf()
        plt.close()'''
        output = pd.DataFrame({
                        "value":y,
                        "trend":res.trend
                    })
        print(OutputPath+fileName)
        output.to_csv( OutputPath+fileName, index=False)
