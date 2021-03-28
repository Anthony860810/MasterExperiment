import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

Folder_PATH = "./A4Benchmark/"
OUTPUT_PATH = "../../anomaly_detection_data/YahooBenchmark/A4Benchmarktrend/"

for fileName in os.listdir(Folder_PATH):
    if(fileName[0:14]=="A4Benchmark-TS"):
        file = os.path.join(Folder_PATH,fileName)
        print(file)
        df = pd.read_csv(file)
        plt.plot(df["value"], label="value")
        plt.plot(df["trend"], label="trend")
        plt.savefig(OUTPUT_PATH+fileName[:-4]+".png")
        plt.clf()
        plt.close()