import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Folder_PATH = "./A4Benchmark/"
OUTPUT_PATH = "../../anomaly_detection_data/YahooBenchmark/A4_LinearRegression/"
x=[]
y=[]
for fileName in os.listdir(Folder_PATH):
    if(fileName[0:14]=="A4Benchmark-TS"):
        file = os.path.join(Folder_PATH,fileName)
        #print(file)
        df = pd.read_csv(file)
        x.append(df["value"])
        y.append(df["trend"])
x = np.array(x)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

model = LinearRegression().fit(x_train,y_train)
y_hat = model.predict(x_test)
for i in range(y_hat.shape[0]):
    print(i)
    plt.plot(x_test[i], label="y_test")
    plt.plot(y_test[i], label="y_test")
    plt.plot(y_hat[i], label="y_hat")
    plt.savefig(OUTPUT_PATH+str(i)+".png")
    plt.clf()
    plt.close()
