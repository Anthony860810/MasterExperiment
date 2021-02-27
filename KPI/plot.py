import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

Folder_PATH = "/home/tony/python_project/anomaly_detection_data/KPI_data/"
OUTPUT_PATH = "/home/tony/python_project/anomaly_detection_data/KPI_plot_pdf/"
i=1
for fileName in os.listdir(Folder_PATH):
    file = os.path.join(Folder_PATH,fileName)
    print(fileName)
    df = pd.read_csv(file)
    label = df["label"]
    anomaly_point = list(np.where(label == 1))
    if len(anomaly_point)!=0:
        plt.plot(df["value"], '-gD', markevery=anomaly_point, linewidth= 0.25, color='#4F9D9D',mfc="yellow", mec="red", markersize=0.3)
        plt.savefig(OUTPUT_PATH+fileName[:-4]+".pdf")
        plt.clf()
        plt.close()