import os
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

Folder_PATH = "/home/tony/python_project/anomaly_detection_data/NAB/data/"
Labels_PATH = "/home/tony/python_project/anomaly_detection_data/NAB/labels/combined_labels.json"
OUTPUT_PATH = "/home/tony/python_project/anomaly_detection_data/NAB_plot_png/"

f = open(Labels_PATH)
file = json.load(f)
for csvname in file:
    print(csvname)
    ans={}
    for point in file[csvname]:
        ans[point]=1
    
    data_path = os.path.join(Folder_PATH, csvname)
    df = pd.read_csv(data_path)
    anomaly_points=[]
    for idx in range (len(df)):
        if df["timestamp"][idx] in ans:
            anomaly_points.append(idx)
    if ans:
        plt.plot(df["value"], '-gD', markevery=anomaly_points, linewidth= 0.25, color='#4F9D9D',mfc="yellow", mec="red", markersize=0.5)
        plt.savefig(OUTPUT_PATH+csvname[:-4]+".png")
        plt.clf()
        plt.close()
    else:
        plt.plot(df["value"],  linewidth= 0.25, color='#4F9D9D')
        plt.savefig(OUTPUT_PATH+csvname[:-4]+".png")
        plt.clf()
        plt.close()
