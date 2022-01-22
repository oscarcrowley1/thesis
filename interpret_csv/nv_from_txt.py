from cmath import nan
import csv
from functools import total_ordering
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import datetime as datetime

read_filename = "interpret_csv/ALPHA/all_junc_alpha_01Jun18.txt"
write_filename = "interpret_csv/ALPHA/new_all_junc.csv"

text_file = open(read_filename, "r")
output_file = open(write_filename, "w")

for line in text_file:
    line = line.replace("^", " ")
    line = line.replace("*", " ")
    line = line.replace("!", " ")
    line = line.replace("'", " ")
    line = line.replace(">", " ")
    line = line.replace("-       -", "-   -   -")
    line.strip()
    line = re.sub("\h+", ",", line)
    
    output_file.write(line)
    
text_file.close()
output_file.close()

df_titles = ["Day", "INT", "OTHER", "SA/LK", "PH", "PT", "DS_A", "VO_A", "VK_A", "DS_B", "VO_B", "VK_B", "DS_C", "VO_C", "VK_C", "DS_D", "VO_D", "VK_D", "ADS"]

df = pd.read_csv(write_filename, names=df_titles, skiprows=1)


print(df.head())
current_time = "00:00"
df_size = df.size
current_time_list = []

for index, row in df.iterrows():
    other_column = str(row["OTHER"])
    if ":" in other_column:
        current_time = other_column
        
    current_time_list.append(current_time)
    
df["Current Time"] = current_time_list
    

print(df.head(25))
print(df.tail(25))

x = (df.iloc[15][2])

print(x)
print(type(x))
if ":" in x:
    print("YES?")
