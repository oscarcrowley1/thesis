from cmath import exp
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


epsilon = 2.5
delta_squared = 10

def gauss_kernel(array, array_copy):
    if(array > epsilon):
        return 0
    return_array = np.exp(-(array*array_copy)/delta_squared)
    return return_array

v_gauss_kernel = np.vectorize(gauss_kernel)

adj_array_prek = np.load("interpret_csv_bravo/distance_mat_bravo.npy")
#df = pd.read_csv("ALPHA/adj_alpha.csv", header=None)
# adj_array_prek = np.loadtxt(adj_file, delimiter=",")

#print(df)
print(adj_array_prek)
plt.imshow(adj_array_prek)
plt.xlabel("To Sensor")
plt.ylabel("From Sensor")
plt.title("Distance Matrix")
plt.colorbar(label="Distance (kilometres)")
plt.show()

adj_array = v_gauss_kernel(adj_array_prek, adj_array_prek)

adj_array.astype(np.float32)

print(adj_array)

#np.save("adj_mat_alpha", adj_array)

#NOT SAVING AS THIS IS THE OLD VERSION

#values = np.unique(adj_array.ravel())

plt.imshow(adj_array)
plt.xlabel("To Sensor")
plt.ylabel("From Sensor")
plt.title("Adjacency Matrix")
plt.colorbar(label="Adjacency (no unit)")
plt.show()
# colors = [ im.cmap(im.norm(value)) for value in values]
# # create a patch (proxy artist) for every color 
# patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
# # put those patched as legend-handles into the legend
# plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
# plt.colorbar()
# #plt.legend()
# plt.show()
