import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt

DataPath = "../YahooBenchmark/A4Benchmark/"
OutputPath = "../../anomaly_detection_data/YahooBenchmark/A4Benchmark_hpfilter_20000/"
for fileName in os.listdir(DataPath):
    if(fileName[0:14]=="A4Benchmark-TS"):
        file = os.path.join(DataPath,fileName)
        print(file)
        df = pd.read_csv(file)

        y = df['value'].to_numpy()
        cycle, trend = sm.tsa.filters.hpfilter(y,20000)
        '''plt.plot(df["value"], label="value")
        plt.plot(trend, label="trend")
        plt.savefig(OutputPath+fileName[:-4]+".png")
        plt.clf()
        plt.close()'''
        output = pd.DataFrame({
                        "value":y,
                        "trend":trend
                    })
        print(OutputPath+fileName)
        output.to_csv( OutputPath+fileName, index=False)
