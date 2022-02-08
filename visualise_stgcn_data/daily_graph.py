import os
import zipfile
import numpy as np
import torch
import matplotlib.pyplot as plt

if (not os.path.isfile("STGCN-PyTorch-master/data/adj_mat_alpha.npy")
            or not os.path.isfile("STGCN-PyTorch-master/data/node_values_alpha.npy")):
        with zipfile.ZipFile("STGCN-PyTorch-master/data/SCATS.zip", 'r') as zip_ref:
            zip_ref.extractall("STGCN-PyTorch-master/data/")
    
A = np.load("STGCN-PyTorch-master/data/interpret_csv/adj_mat_alpha.npy")
A = A.astype(np.float32)
X = np.load("STGCN-PyTorch-master/data/interpret_csv/node_values_alpha.npy").transpose((1, 2, 0))
X = X.astype(np.float32)

one_station_channel = X[0, 0, :]

print(one_station_channel)


for i in range(int(len(one_station_channel) / 1440)):
    print(one_station_channel[i*1440:(i+1)*1440])
    one_day = one_station_channel[i*1440:(i+1)*1440]
    plt.plot(one_day)
    plt.show()
    
plt.show()