import os
from typing import MutableSequence
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

Folder_PATH = "../YahooBenchmark/A4Benchmark_hpfilter_20000/"
OutputPath = "../YahooBenchmark/A4Benchmark_l1norm_20000_png/"
def Evaluation(y, y_hat):
    print("MSE: "+ str(mean_squared_error(y, y_hat)))
    print("MAE: "+ str(mean_absolute_error(y, y_hat)))
    print("R2_score: "+ str(r2_score(y, y_hat)))


if __name__ == '__main__':
    x=[]
    y=[]
    for fileName in os.listdir(Folder_PATH):
        if(fileName[0:14]=="A4Benchmark-TS"):
            file = os.path.join(Folder_PATH,fileName)
            df = pd.read_csv(file)
            x.append(df["value"])
            y.append(df["trend"])
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    model = LinearRegression().fit(x_train,y_train)
    y_hat = model.predict(x_test)
    for i in range(len(y_hat)):
        plt.plot(x_test[i], linewidth=1.0, color="blue")
        plt.plot(np.array(y_test[i]), linewidth=0.7, color="#D9B300")
        plt.plot(np.array(y_hat[i]), linewidth=0.7, color="red")
        plt.savefig(OutputPath+str(i)+".png")
        plt.clf()
        plt.close()
    Evaluation(y_test, y_hat)