import os
import zipfile
import numpy as np
import torch
import matplotlib.pyplot as plt

experiment = "alpha"
# if (not os.path.isfile("DC_STGCN/data/adj_mat_alpha.npy")
#             or not os.path.isfile("DC_STGCN/data/node_values_alpha.npy")):
with zipfile.ZipFile("DC_STGCN/data/SCATS.zip", 'r') as zip_ref:
    zip_ref.extractall("DC_STGCN/data/")
    
A = np.load("DC_STGCN/data/interpret_csv/adj_mat_alpha.npy")
A = A.astype(np.float32)
X = np.load("interpret_csv/node_values_alpha.npy").transpose((1, 2, 0))
X = X.astype(np.float32)

one_station_channel = X[0, 0, :]

print(one_station_channel)

day_length = 480

for i in range(int(len(one_station_channel) / 480)):
    print(one_station_channel[i*480:(i+1)*480])
    one_day = one_station_channel[i*480:(i+1)*480]
    plt.plot(one_day)
    plt.show()
    
plt.show()