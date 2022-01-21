from cmath import nan
import csv
from functools import total_ordering
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from torch import float32, float64

df = pd.read_csv("ALPHA/all_junc_alpha_01Jun18.csv", header=1)

print(df)

used_count = 0
unused_count = 0

def fill_array(array):
    # for sensor_column in range(len(array[0,:])):
    #     column_values = []
    #     for time_row in range(len(array[:,0])):
    #         current_value = array[time_row, sensor_column]
            
    #         if(math.isnan(current_value)):
    #             column_values.append(-1)
    #         else:
    #             column_values.append(current_value)
    #     plt.plot(column_values)
    # plt.show()
    
    last_value = nan
    current_value = nan
    next_value = nan
    non_nan_count = 0
    nan_count = 0

    for sensor_column in range(len(array[0,:])):
        for time_row in range(len(array[:,0])):
            current_value = array[time_row, sensor_column]
            if(math.isnan(current_value)):  #if current is nan
                if(time_row+1 >= len(array[:,0])):   #if checking beyond the limit take last
                    array[time_row, sensor_column] = last_value
                    break
                else:
                    next_value = array[time_row+1, sensor_column]   #otherwise next value is checked

                nan_count = nan_count+1
                num_steps = 1
                while(math.isnan(next_value)):  #if next value is nan then we keep stepping on
                    num_steps = num_steps+1
                    if(time_row+num_steps >= len(array[:,0])):   #if checking beyond limit
                        array[time_row, sensor_column] = last_value
                        break
                    else:
                        next_value = array[time_row+num_steps, sensor_column]   #otherwise our new values is num_steps on

                if(math.isnan(last_value)): #if it has only been nans until now we take the next value
                    array[time_row, sensor_column] = next_value
                else:   #otherwise we take the weighted average towards last step
                    array[time_row, sensor_column] = (last_value*num_steps + next_value)/(1+num_steps)

            else:
                last_value = array[time_row, sensor_column]
                non_nan_count = non_nan_count+1
    # print("BEFORE")
    # print(f"Non: {non_nan_count}\tNan: {nan_count}\tSum: {nan_count+non_nan_count}")

    # non_nan_count = 0
    # nan_count = 0

    # for sensor_column in range(len(array[0,:])):
    #     column_values = []
    #     for time_row in range(len(array[:,0])):
    #         current_value = array[time_row, sensor_column]
            
    #         if(math.isnan(current_value)):
    #             nan_count = nan_count+1
    #             column_values.append(-1)
    #         else:
    #             non_nan_count = non_nan_count+1
    #             column_values.append(current_value)
    #     plt.plot(column_values)
    # plt.show()

    # print("AFTER")
    # print(f"Non: {non_nan_count}\tNan: {nan_count}\tSum: {nan_count+non_nan_count}")

    return array






def sensor_to_index(junc_num, sensor_letter):
    """[summary]

    Args:
        junc_num ([type]): [description]
        sensor_letter ([type]): [description]

    Returns:
        [type]: [description]
    """
    if(junc_num=="183"):
        if(sensor_letter=="AC"):
            return int(0)
        elif(sensor_letter=="B"):
            return int(1)
        else:
            print("ERROR IN S_TO_I, JUNC: " + str(junc_num) + "SENS: " + str(sensor_letter))
            return int(0)   
    elif(junc_num=="196"):
        if(sensor_letter=="A"):
            return int(2)
        elif(sensor_letter=="ABE"):
            return int(3)
        elif(sensor_letter=="D"):
            return int(4)
        else:
            print("ERROR IN S_TO_I, JUNC: " + str(junc_num) + "SENS: " + str(sensor_letter))
            return int(0)
    elif(junc_num=="288"):
        if(sensor_letter=="A"):
            return int(5)
        elif(sensor_letter=="B"):
            return int(6)
        else:
            print("ERROR IN S_TO_I, JUNC: " + str(junc_num) + "SENS: " + str(sensor_letter))
            return int(0)
    elif(junc_num=="354"):
        if(sensor_letter=="A"):
            return int(7)
        elif(sensor_letter=="B"):
            return int(8)
        else:
            print("ERROR IN S_TO_I, JUNC: " + str(junc_num) + "SENS: " + str(sensor_letter))
            return int(0)
    else:
        print("ERROR IN S_TO_I, JUNC: " + str(junc_num) + "SENS: " + str(sensor_letter))
        return int(0)

def time_to_index(time_value):
    """[summary]

    Args:
        time_value ([type]): [description]

    Returns:
        [type]: [description]
    """
    h, m, s = time_value.split(":", 3)
    h = int(h)
    m = int(m)
    #print(f"H: {h}, M: {m}, S: {s}", end='\r')
    # print(f"H={h}")
    # print(f"M={m}")
    return int((h*60) + m)

time_dim = 1440
sensor_dim = 9
channel_dim = 2

# output_array = np.empty((time_dim, sensor_dim, channel_dim))
# output_array[:] = np.NaN

density_array = np.empty((time_dim, sensor_dim))
density_array[:] = np.NaN

flow_array = np.empty((time_dim, sensor_dim))
flow_array[:] = np.NaN

# print(output_array.shape)
# print(output_array[0, 0, 0])


for index, row in df.iterrows():
    if(row['Remove Duplicate']==1):
        used_count = used_count+1
        sensor_index = sensor_to_index(row["INT"], row["PH"])
        time_index = time_to_index(row["Current Time"])
        #print(f"TIMETYPE: {type(time_index)} SENSORTYPE: {type(sensor_index)}")
        try:
            sensor_index = int(sensor_index)
            time_index = int(time_index)
        except:
            print("NOT INTEGER")
        total_vo = int(row["Total VO"])
        avg_ds = int(row["Average DS"])
        phase_time = int(row["PT"])
        t_small = 1

        if(phase_time != 0):
            flow = total_vo*60/phase_time
            t_bign = phase_time*(1-avg_ds) - (t_small * total_vo)
            density = (1-t_bign)/phase_time
        else:
            flow = 0
            t_bign = 0
            density = 0

        print(f"TOTAL_VO: {total_vo}")

        print(time_index)
        flow_array[time_index, sensor_index] = flow
        density_array[time_index, sensor_index] = density

    else:
        unused_count = unused_count+1

#fill_array(flow_array)
print(len(flow_array[0,:]))
print(len(flow_array[:,0]))

old_flow_array = flow_array
old_density_array = density_array

flow_array = fill_array(flow_array)
density_array = fill_array(density_array)



print(f"Used: {used_count}\tUnused: {unused_count}\tSum: {used_count+unused_count}")

output_array = np.stack([flow_array, density_array], axis=2)

print(output_array.shape) # 1440*9*2

#print(output_array.type)
output_array.astype(np.float64)
# output_array.astype(float32)

np.save("node_values_alpha", output_array)
