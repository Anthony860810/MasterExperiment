import os
import pandas as pd
FolderPath = "D:/pythonspace/Anomaly detection/data"
FileList = os.listdir(FolderPath)


files_days = []
for file in FileList:
    file_path = FolderPath+"/"+file
    df = pd.read_csv(file_path)
    files_days.append(len(df))

sorted_files= sorted(files_days)
txt = open("DataDaysCalculate.txt", mode="w")
for day in sorted_files:
    txt.write(str(day)+"\n")
txt.close()