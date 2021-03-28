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

Folder_PATH = "../YahooBenchmark/A4Benchmark/"

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
    Evaluation(y_test, y_hat)