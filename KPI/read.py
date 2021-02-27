import pandas as pd
import numpy as np

Output_Path = "KPIData/"
data = pd.read_csv("KPIData/phase2_train.csv")
categories = pd.unique(data["KPI ID"])
print(categories)
print(len(categories))
dataset = {}
by_class = data.groupby("KPI ID")

for groups, data in by_class:
    dataset[groups] = data

for filename in categories:
    df = dataset[filename]
    df.to_csv( Output_Path+filename + ".csv", index=False)