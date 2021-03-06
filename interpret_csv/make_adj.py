from cmath import exp
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


epsilon = 1
delta_squared = 10

def gauss_kernel(array, array_copy):
    if(array > epsilon):
        return 0
    return_array = np.exp(-(array*array_copy)/delta_squared)
    return return_array

v_gauss_kernel = np.vectorize(gauss_kernel)

adj_file = open("interpret_csv/ALPHA/adj_alpha.csv")
#df = pd.read_csv("ALPHA/adj_alpha.csv", header=None)
adj_array_prek = np.loadtxt(adj_file, delimiter=",")

#print(df)
print(adj_array_prek)

adj_array = v_gauss_kernel(adj_array_prek, adj_array_prek)

print(adj_array)

adj_array.astype(np.float32)

print(adj_array)

#np.save("adj_mat_alpha", adj_array)

#NOT SAVING AS THIS IS THE OLD VERSION

#values = np.unique(adj_array.ravel())

im = plt.imshow(adj_array)
# colors = [ im.cmap(im.norm(value)) for value in values]
# # create a patch (proxy artist) for every color 
# patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
# # put those patched as legend-handles into the legend
# plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.colorbar()
#plt.legend()
plt.show()
