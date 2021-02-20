from RobustSTL import RobustSTL
from sample_generator import sample_generation
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
##test_error
Data_Path = "D:/pythonspace/Anomaly detection/data/test_error.csv"
Plot_Path = "D:/pythonspace/Anomaly detection/prophet_changepoint_plot/"

def main(file):
    data = pd.read_csv(file)
    data = data["value"].to_numpy()
    norm1 = data / np.linalg.norm(data)
    result = RobustSTL(norm1, 5, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)
    forecasting(file, data, result)
def forecasting(file, OriginalData, Rstl_result):
    model = Prophet()
    data = pd.read_csv(file)
    '''
    Rstl_result[0]: original data
    Rstl_result[1]: trend
    Rstl_result[2]: seasonality
    Rstl_result[3]: remainder
    '''
    trend = pd.DataFrame({  "ds": data["timestamp"],
                            "y":Rstl_result[1]
                        })
    model.fit(trend)
    #future = model.make_future_dataframe(periods=30)
    #forecast = model.predict(future)
    #fig = model.plot(forecast)
    #a = add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.plot(Rstl_result[1], '-gD', markevery=model.changepoints.index, linewidth= 0.25, color='#7A0099',mfc="yellow", mec="red", markersize=0.3)
    plt.savefig(Plot_Path +"test_error_Prophet_ByTrend_AnomalyDetection_trend.pdf")
    plt.clf()
    plt.close()
if __name__ == '__main__':
    main(Data_Path)