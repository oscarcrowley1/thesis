import numpy as np
import matplotlib.pyplot as plt

array = np.load("DC-STGCN/data/interpret_csv/adj_mat_alpha.npy")

plt.imshow(array)
plt.xlabel("To Sensor")
plt.ylabel("From Sensor")
plt.title("Pre-Kernelisation Adjacency Matrix")
plt.colorbar()
plt.show()