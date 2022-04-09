import numpy as np
import matplotlib.pyplot as plt

adjacency = np.load("DC-STGCN/data/adj_mat.npy")
print(adjacency.dtype)

adjacency_alpha = np.load("DC-STGCN/data/adj_mat_alpha.npy")
print(adjacency_alpha.dtype)

node_vals = np.load("DC-STGCN/data/node_values.npy")
print(node_vals.dtype)

node_vals_alpha = np.load("DC-STGCN/data/node_values_alpha.npy")
print(node_vals_alpha.dtype)

sheet = 0

print(node_vals[:, :, sheet])

node_range = range(207)
time_range = range(34272)

# for i in range(207):
#     plt.plot(time_range, node_vals[:, i, sheet])


# plt.show()
A = adjacency
X = node_vals.transpose((1, 2, 0))
X = X.astype(np.float32)

# Normalization using Z-score method
means = np.mean(X, axis=(0, 2))
X = X - means.reshape(1, -1, 1)
stds = np.std(X, axis=(0, 2))
X = X / stds.reshape(1, -1, 1)

print(means)

for i in range(207):
    plt.plot(time_range, X[i, sheet, :])

plt.show()


print("WORKING?")
#return A, X, means, stds