from RobustSTL import RobustSTL
from sample_generator import sample_generation
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

Data_Path = "D:/pythonspace/Anomaly detection/data"
Plot_Path = "D:/pythonspace/Anomaly detection/rstl_plot/"

def main(folder):
    for fileName in os.listdir(folder):
        file = os.path.join(folder,fileName)
        print(file)
        data = pd.read_csv(file)
        data = data["value"].to_numpy()
        norm1 = data / np.linalg.norm(data)
        print(norm1)
        ##result = RobustSTL(sample_list[0], 50, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)
        result = RobustSTL(norm1, 50, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)
        fig = plt.figure(figsize=(30,25))
        matplotlib.rcParams.update({'font.size': 22})
        samples = zip(result, ['sample', 'trend', 'seasonality', 'remainder'])

        for i, item in enumerate(samples):
            plt.subplot(4,1,(i+1))
            if i==0:
                plt.plot(item[0], color='blue')
                plt.title(item[1])
                plt.subplot(4,1,i+2)
                plt.plot(item[0], color='blue')
            else:
                plt.plot(item[0], color='red')
                plt.title(item[1])
        plt.savefig(Plot_Path + fileName[:-4]+".pdf")
        plt.clf()
        plt.close()
    
if __name__ == '__main__':
    main(Data_Path)