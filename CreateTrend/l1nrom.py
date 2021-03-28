import pandas as pd
import numpy as np
import cvxpy as cp
import scipy
import cvxopt
import matplotlib.pyplot as plt
import os

DataPath = "../YahooBenchmark/A4Benchmark/"
OutputPath = "../../anomaly_detection_data/YahooBenchmark/A4Benchmark_l1norm_20000/"


for fileName in os.listdir(DataPath):
    if(fileName[0:14]=="A4Benchmark-TS"):
        file = os.path.join(DataPath,fileName)
        print(file)
        df = pd.read_csv(file)
        y = df['value'].to_numpy()

        ones_row = np.ones((1, len(y)))
        D = scipy.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), len(y)-2, len(y))
        vlambda = 20000
        y_hat = cp.Variable(shape=len(y))

        objective_func = cp.Minimize(0.5*cp.sum_squares(y-y_hat)+vlambda*cp.norm(D@y_hat, 1))
        problem = cp.Problem(objective_func)
        problem.solve(verbose=True)
        print(np.array(y_hat.value))
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        '''plt.plot(y, linewidth=1.0, color="blue")
        plt.plot(np.array(y_hat.value), linewidth=0.5, color="red")
        plt.savefig(OutputPath+fileName[:-4]+".png")
        plt.clf()
        plt.close()'''
        output = pd.DataFrame({
                        "value":y,
                        "trend":y_hat
                    })
        output.to_csv( OutputPath+fileName, index=False)