import numpy as np
import pandas as pd
#import folium
import webbrowser
import os
import math
import matplotlib.pyplot as plt
import zipfile


#from h3 import h3
# import folium
# from folium import Map
# from folium import plugins

import openrouteservice 
#from openrouteservice import convert
import json

epsilon = 0.6
delta_squared = 10 # oringinally 10

def gauss_kernel(array, array_copy):
    if(array > epsilon):
        return 0
    return_array = np.exp(-(array*array_copy)/delta_squared)
    return return_array

v_gauss_kernel = np.vectorize(gauss_kernel)

client = openrouteservice.Client(key='5b3ce3597851110001cf62484b7969c841cd4ddab8eb2dc7e1f53738')
#res = client.directions(coords)
#set location coordinates in longitude,latitude order
#coords = ((-6.240513, 53.361145),(-6.292406, 53.355797))

np.set_printoptions(suppress=True) # stops scientific notation

adj_array = np.diag([1,1,1,1,1,1,1,1,1])
print(adj_array)

plt.imshow(adj_array)
plt.xlabel("To Sensor")
plt.ylabel("From Sensor")
plt.title("Pre-Kernelisation Adjacency Matrix")
plt.colorbar()
plt.show()



#commmented when not saving
np.save("interpret_csv/adj_mat_alpha", adj_array)


if os.path.isfile("interpret_csv/node_values_alpha.npy") and os.path.isfile("interpret_csv/nv_info.txt"):
    with zipfile.ZipFile("interpret_csv/SCATS_alpha.zip", "w") as zip_object:
        zip_object.write("interpret_csv/node_values_alpha.npy")
        zip_object.write("interpret_csv/adj_mat_alpha.npy")
        zip_object.write("interpret_csv/adj_info.txt")
        zip_object.write("interpret_csv/nv_info.txt")
    print("Zipped")