from RobustSTL import RobustSTL
from sample_generator import sample_generation
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

Data_Path = "D:/pythonspace/Anomaly detection/data"
Plot_Path = "D:/pythonspace/Anomaly detection/rstl_outlier/"

def main(folder):
    for fileName in os.listdir(folder):
        if fileName!="test_error.csv":
            file = os.path.join(folder,fileName)
            print(file)
            data = pd.read_csv(file)
            data = data["value"].to_numpy()
            norm1 = data / np.linalg.norm(data)
            result = RobustSTL(norm1, 50, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)
            forecasting(fileName, file, data, result)
def forecasting(fileName, file, OriginalData, Rstl_result):
    abs_remainder = np.abs(Rstl_result[3])
    data = pd.read_csv(file)
    threshold = np.percentile(abs_remainder , 95)
    anomaly_point = list(np.where(abs_remainder > threshold))
   
    '''
    Rstl_result[0]: original data
    Rstl_result[1]: trend
    Rstl_result[2]: seasonality
    Rstl_result[3]: remainder
    '''
 
    plt.plot(Rstl_result[0], '-gD', markevery=anomaly_point, linewidth= 0.25, color='#7A0099',mfc="yellow", mec="red", markersize=0.3)
    plt.savefig(Plot_Path + fileName[:-4]+"_rstl_outlier.pdf")
    plt.clf()
    plt.close()
if __name__ == '__main__':
    main(Data_Path)