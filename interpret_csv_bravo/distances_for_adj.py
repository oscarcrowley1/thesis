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
delta_squared = 10 # oringinally 10, 1 gave good spread

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

coord_array = np.array([[53.347369, -6.255022], #48i
                        [53.347474, -6.254784], #48ii
                        [53.346772, -6.259125], #48iii
                        [53.347122, -6.258851], #145i
                        [53.347054, -6.258609], #145ii
                        [53.344705, -6.252184], #145iii
                        [53.344835, -6.252252], #152i
                        [53.349220, -6.251510], #152ii
                        [53.349175, -6.251907]  #152iii
                        [53.349175, -6.251907]  #354i
                        [53.349175, -6.251907]  #354ii
                        ])

#coord_array = np.ndarray(coord_array)
num_points = coord_array.shape[0]
adj_array = np.empty([num_points, num_points])

for from_point in range(num_points):
    for to_point in range(num_points):
        if (to_point != from_point):
            coords = ((coord_array[from_point][1], coord_array[from_point][0]), (coord_array[to_point][1], coord_array[to_point][0]))
            
            #print(coords)
            #coords = ((-6.252248, 53.344848),(-6.255019, 53.347365))
            print(coords)

            #call API
            res = client.directions(coords)
            #test our response
            # with(open('test.json','+w')) as f:
            #     f.write(json.dumps(res,indent=4, sort_keys=True))

            # geometry = client.directions(coords)['routes'][0]['geometry']
            # decoded = convert.decode_polyline(geometry)
            # print(decoded)

            #folium.GeoJson(decoded).add_to(fgRoute)

            #distance_txt = "<h4> <b>Distance :&nbsp" + "<strong>"+str(res['routes'][0]['summary']['distance']/1000)+" Km </strong>" +"</h4></b>"
            #duration_txt = "<h4> <b>Duration :&nbsp" + "<strong>"+str(round(res['routes'][0]['summary']['duration']/60,1))+" Mins. </strong>" +"</h4></b>"
            
            distance = round(res['routes'][0]['summary']['distance']/1000, 4)
            
            
            
        else:
            distance = 0
            
        print(f"DISTANCE from point {from_point} to {to_point} is:\t{distance}")
        print("\n")
        adj_array[from_point][to_point] = distance
            
        
            
print(adj_array)

plt.imshow(adj_array)
plt.xlabel("To Sensor")
plt.ylabel("From Sensor")
plt.title("Pre-Kernelisation Adjacency Matrix")
plt.colorbar()
plt.show()

kernelised_adj = v_gauss_kernel(adj_array, adj_array)

print(kernelised_adj)

plt.imshow(kernelised_adj)
plt.xlabel("To Sensor")
plt.ylabel("From Sensor")
plt.title("Post-Kernelisation Adjacency Matrix")
plt.colorbar()
plt.show()

## TRYING TRASPOSE
kernelised_adj = np.transpose(kernelised_adj)

##commmented when not saving
np.save("interpret_csv_bravo/adj_mat_bravo", kernelised_adj)

f = open("interpret_csv_bravo/adj_info.txt", "w")
info_string = "Epsilon:\t" + str(epsilon) + "\nDelta Squared:\t" + str(delta_squared) + "\nUses these distances\n" + str(coord_array)
f.write(info_string)
f.close()


if os.path.isfile("interpret_csv_bravo/node_values_bravo.npy") and os.path.isfile("interpret_csv_bravo/nv_info.txt"):
    with zipfile.ZipFile("interpret_csv_bravo/SCATS_bravo.zip", "w") as zip_object:
        zip_object.write("interpret_csv_bravo/node_values_bravo.npy", arcname="bravo_data/node_values_bravo.npy")
        zip_object.write("interpret_csv_bravo/adj_mat_bravo.npy", arcname="bravo_data/adj_mat_bravo.npy")
        zip_object.write("interpret_csv_bravo/adj_info.txt", arcname="bravo_data/adj_info.npy")
        zip_object.write("interpret_csv_bravo/nv_info.txt", arcname="bravo_data/nv_info.npy")
    print("Zipped")

