import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

Folder_PATH = "./A4Benchmark/"
OUTPUT_PATH = "D:/pythonspace/Anomaly detection/YahooBenchmark/A4Benchmarkpng/"
i=1
for fileName in os.listdir(Folder_PATH):
    if(fileName[0:14]=="A4Benchmark-TS"):
        file = os.path.join(Folder_PATH,fileName)
        print(file)
        df = pd.read_csv(file)
        label = df["anomaly"]
        anomaly_point = list(np.where(label == 1))
        if len(anomaly_point)!=0:    
            plt.plot(df["value"], '-gD', markevery=anomaly_point, linewidth= 0.25, color='#4F9D9D',mfc="yellow", mec="red", markersize=0.3)
            plt.savefig(OUTPUT_PATH+fileName[:-4]+".png")
            plt.clf()
            plt.close()