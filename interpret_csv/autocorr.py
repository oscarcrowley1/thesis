import os
import zipfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from pytorch_forecasting.utils import autocorrelation

# if (not os.path.isfile("STGCN-PyTorch-master/data/adj_mat_alpha.npy")
#             or not os.path.isfile("STGCN-PyTorch-master/data/node_values_alpha.npy")):
#         with zipfile.ZipFile("STGCN-PyTorch-master/data/SCATS.zip", 'r') as zip_ref:
#             zip_ref.extractall("STGCN-PyTorch-master/data/")

# if (not os.path.isfile("adj_mat_alpha.npy")
#             or not os.path.isfile("node_values_alpha.npy")):
#         with zipfile.ZipFile("SCATS.zip", 'r') as zip_ref:
#             zip_ref.extractall("STGCN-PyTorch-master/data/")
    
A = np.load("adj_mat_alpha.npy")
A = A.astype(np.float32)
X = np.load("node_values_alpha.npy").transpose((1, 2, 0))
X = X.astype(np.float32)
X = torch.Tensor(X)
output = autocorrelation(X, dim=2)
print(X.shape)
print(output.shape)

one_station_channel = output[:, 0, :]
print(one_station_channel.shape)
for station_num in range(one_station_channel.shape[0]):
    plt.plot(one_station_channel[station_num, :], label=f"Station {station_num}")

plt.axvline(x=5)
plt.axvline(x=480)
plt.axvline(x=480*7)

plt.legend()
plt.show()

#output2 = autocorrelation(X[:, :, 10000:], dim=2)
#one_station_channel = output2[:, 0, :]
#print(one_station_channel.shape)
output = output.cpu().detach().numpy()

for station_num in range(one_station_channel.shape[0]):
    oned_data = one_station_channel[station_num, :]
    indices = torch.Tensor(argrelextrema(oned_data, np.greater))
    plt.plot(oned_data[indices], indices, label=f"Station {station_num}")

plt.axvline(x=5)
plt.axvline(x=480)
plt.axvline(x=480*7)

plt.legend()
plt.show()
# print(one
# 
# _station_channel)

# day_length = 480

# for i in range(int(len(one_station_channel) / 480)):
#     print(one_station_channel[i*480:(i+1)*480])
#     one_day = one_station_channel[i*480:(i+1)*480]
#     plt.plot(one_day)
#     plt.show()
    
# plt.show()