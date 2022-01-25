from cmath import nan
import csv
from functools import total_ordering
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
from datetime import datetime
import os
import zipfile

def time_to_index(time_value, first_date, date_obj):
    h, m = time_value.split(":", 2)
    # date_diff = date_obj - first_date
    # d = date_diff.days # DOESNT NEED THIS AS ARRAYS ARE MADE ONE AT A TIME
    h = int(h)
    m = int(m)
    return int((h*60) + m)


def sensor_to_index(junc_num, sensor_letter):
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

def fill_array(array):
    # for sensor_column in range(len(array[0,:])):
    #     column_values = []
    #     for time_row in range(len(array[:,0])):
    #         current_value = array[time_row, sensor_column]
            
    #         if(math.isnan(current_value)):
    #             column_values.append(100)
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
    print("BEFORE")
    print(f"Non: {non_nan_count}\tNan: {nan_count}\tSum: {nan_count+non_nan_count}")

    non_nan_count = 0
    nan_count = 0

    # for sensor_column in range(len(array[0,:])):
    #     column_values = []
    #     for time_row in range(len(array[:,0])):
    #         current_value = array[time_row, sensor_column]
            
    #         if(math.isnan(current_value)):
    #             nan_count = nan_count+1
    #             column_values.append(100)
    #         else:
    #             non_nan_count = non_nan_count+1
    #             column_values.append(current_value)
    #     plt.plot(column_values)
    # plt.show()

    print("AFTER")
    print(f"Non: {non_nan_count}\tNan: {nan_count}\tSum: {nan_count+non_nan_count}")

    return array

txt_directory = "interpret_csv/ALPHA_1week"
csv_directory = "interpret_csv/csv_files"

concat_flow_array = None
concat_density_array = None

first_date = None

for txt_filename in sorted(os.listdir(txt_directory)):

    # read_filename = "interpret_csv/ALPHA/all_junc_alpha_01Jun18.txt"
    # write_filename = "interpret_csv/ALPHA/new_all_junc.csv"
    name, suffix = txt_filename.split(".", 2)
    csv_filename = name + ".csv"


    read_filename = os.path.join(txt_directory, txt_filename)
    write_filename = os.path.join(csv_directory, csv_filename)

    text_file = open(read_filename, "r")
    output_file = open(write_filename, "w")

    for line in text_file:
        line = line.replace("^", " ")
        line = line.replace("*", " ")
        line = line.replace("!", " ")
        line = line.replace("'", " ")
        line = line.replace(">", " ")
        line = line.replace("-       -", "-   -   -")
        line.strip()
        line = re.sub("\h+", ",", line)
        
        output_file.write(line)
        
    text_file.close()
    output_file.close()

    df_titles = ["Day", "INT", "OTHER", "SA/LK", "PH", "PT", "DS_A", "VO_A", "VK_A", "DS_B", "VO_B", "VK_B", "DS_C", "VO_C", "VK_C", "DS_D", "VO_D", "VK_D", "ADS"]

    df = pd.read_csv(write_filename, names=df_titles, skiprows=1)

    our_nodes = ["183", "196", "288", "354"]

    print(df.head())
    current_time = "00:00"

    df_size = df.size
    current_time_list = []
    density_list = []
    flow_list = []

    used_count = 0
    unused_count = 0

    time_dim = 1440
    sensor_dim = 9
    channel_dim = 2

    density_array = np.empty((time_dim, sensor_dim))
    density_array[:] = np.NaN

    flow_array = np.empty((time_dim, sensor_dim))
    flow_array[:] = np.NaN

    current_day = df.iloc[0][1]
    current_datetime = datetime.strptime(current_day, "%d-%B-%Y")
    print(current_datetime)

    if first_date is None:
        first_date = current_datetime

    for index, row in df.iterrows():
        other_column = str(row["OTHER"])
        salk_column = str(row["SA/LK"])
        int_column = str(row["INT"])
        phase_time = (row["PT"])
        sensor_column = (row["PH"])

        t_small = 1

        ds_list = [row["DS_A"], row["DS_B"], row["DS_C"], row["DS_D"]]
        vo_list = [row["VO_A"], row["VO_B"], row["VO_C"], row["VO_D"]]
        vk_list = [row["VK_A"], row["VK_B"], row["VK_C"], row["VK_D"]]

        if ":" in other_column:
            current_time = other_column        
        current_time_list.append(current_time)

        check_is_node = salk_column.isnumeric()
        check_node_list = (int_column in our_nodes) and check_is_node

        #check_is_duplicate = (if)
        if check_node_list:
            num_lanes = ((ds_list[0]).isnumeric()) + ((ds_list[1]).isnumeric()) + ((ds_list[2]).isnumeric()) + ((ds_list[3]).isnumeric())
        else:
            num_lanes = None

        check_is_duplicate = check_node_list and (int_column == "183" and (num_lanes == 1 or num_lanes == 2))

        if(check_node_list and not check_is_duplicate):
            used_count = used_count+1
            total_ds = 0
            total_vo = 0
            total_vk = 0

            for lane in range(num_lanes):
                total_ds = total_ds + int(ds_list[lane])
                total_vo = total_vo + int(vo_list[lane])
                total_vk = total_vk + int(vk_list[lane])
            
            
            avg_ds = total_ds/num_lanes
            phase_time = int(phase_time)

            if(phase_time != 0):
                flow = total_vo*60/phase_time
                t_bign = phase_time*(1-avg_ds) - (t_small * total_vo)
                density = (1-t_bign)/phase_time
            else:
                flow = 0
                t_bign = 0
                density = 0
            
            flow_list.append(flow)
            density_list.append(density)

            time_index = time_to_index(current_time, first_date, current_datetime)
            sensor_index = sensor_to_index(int_column, sensor_column)

            flow_array[time_index, sensor_index] = flow
            density_array[time_index, sensor_index] = density


        else:
            flow_list.append(None)
            density_list.append(None)
            unused_count = unused_count+1

    print(f"Used: {used_count}\tUnused: {unused_count}\tSum: {used_count+unused_count}")


        
    df["Current Time"] = current_time_list
    df["Flow"] = flow_list
    df["Density"] = density_list

    flow_array = fill_array(flow_array)
    density_array = fill_array(density_array)
        
    if concat_flow_array is None:
        concat_flow_array = flow_array
    else:
        concat_flow_array = np.concatenate([concat_flow_array, flow_array], axis=0)
        print(concat_flow_array.shape)
        print(flow_array.shape)
        print("STACK")

    if concat_density_array is None:
        concat_density_array = density_array
    else:
        concat_density_array = np.concatenate([concat_density_array, density_array], axis=0)

    print(df.head(25))
    print(df.tail(25))

output_array = np.stack([concat_flow_array, concat_density_array], axis=2)

print(output_array.shape) # 1440*9*2

#print(output_array.type)
output_array.astype(np.float64)
# output_array.astype(float32)

np.save("interpret_csv/node_values_alpha", output_array)

if os.path.isfile("interpret_csv/adj_mat_alpha.npy"):
    with zipfile.ZipFile("interpret_csv/SCATS.zip", "w") as zip_object:
        zip_object.write("interpret_csv/node_values_alpha.npy")
        zip_object.write("interpret_csv/adj_mat_alpha.npy")
