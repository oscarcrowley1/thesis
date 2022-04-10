import os
import zipfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from pytorch_forecasting.utils import autocorrelation

X = np.load("interpret_csv/node_values_alpha.npy").transpose((1, 2, 0))
# X = np.load("interpret_csv_bravo/node_values_bravo.npy").transpose((1, 2, 0))
# X = np.load("interpret_csv_bravoplus/node_values_bravoplus.npy").transpose((1, 2, 0))
X = X.astype(np.float32)
X = torch.Tensor(X)
output = autocorrelation(X, dim=2)
print(X.shape)
print(output.shape)

one_station_channel = output[:, 1, :]
print(one_station_channel.shape)
for station_num in range(one_station_channel.shape[0]):
    plt.plot(np.array(range(one_station_channel.shape[1]))/20, one_station_channel[station_num, :], label=f"Station {station_num}")

plt.axvline(x=5/20, alpha=0.5, c='r', ls='--', label="t")
plt.axvline(x=480/20, alpha=0.5, c='g', ls='--', label="t-(480)+5")
plt.axvline(x=480*7/20, alpha=0.5, c='b', ls='--', label="t-(480*7)+5")
plt.axhline(y=0, alpha=0.1, c='k')

plt.title("Autocorrelation Plot for Junction Set Alpha")
plt.xlabel("Time Difference (hours)")
plt.ylabel("Autocorrelation")
plt.legend(loc='upper right')
plt.show()