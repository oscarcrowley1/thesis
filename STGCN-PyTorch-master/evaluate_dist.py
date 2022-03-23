import torch
import matplotlib.pyplot as plt

mean = 10
std = 0.5

norm_dist = torch.distributions.Normal(mean, std)

check_mean = norm_dist.mean
check_std = norm_dist.stddev

print(f"MEAN:\t{check_mean}\tSTD:\t{check_std}")

from cProfile import label
import os
import argparse
import pickle as pk
from re import S
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from time import process_time
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import sys


from stgcn import STGCN
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save, new_generate_dataset

dist_bool = True

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    print("\nPARAMETERS:")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print_save(f, f"{name}:\t{parameter}")
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':
    A, X, means, stds, info_string = load_scats_data()


    num_timesteps_input = 25
    
    if dist_bool:
        num_timesteps_output = 2
    else:
        num_timesteps_output = 1

    #print_save(f, A)

    ex_split_line1 = int(X.shape[2] * 0.8)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)

    
    total_input, total_target, num_timesteps_input = new_generate_dataset(X)

    ex_test_input = total_input[ex_split_line1:, :, :]
    ex_test_target = total_target[ex_split_line1:, :, :]

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)#removed to device

    ex_net = STGCN(A_wave.shape[0],
                ex_test_input.shape[3],
                num_timesteps_input,
                num_timesteps_output)#.to(device=args.device)
    
    if False:#if torch.cuda.is_available():
        ex_net.load_state_dict(torch.load("saved_models/model_0222_1341_e299"))
    else:
        # ex_net.load_state_dict(torch.load("saved_models/model_0222_1341_e299", map_location=torch.device('cpu')))#for use on my computer
        ex_net.load_state_dict(torch.load("saved_models/model_0323_0634_e999_out2", map_location=torch.device('cpu')))#for use on my computer
    
    with torch.no_grad():
        ex_net.eval()
    
        out = ex_net(A_wave, ex_test_input)
        # print(ex_test_input.shape)
        # print(ex_test_target.shape)
        # print(out.shape)
        
        stop_num = 0
        time_step = 0
        
        ex_test_target_UN = ex_test_target*stds[0]+means[0]
        out_UN_mean = out[:, stop_num, 0]*stds[0]+means[0]
        out_UN_std = out[:, stop_num, 1]*stds[0]
        
        print(out.shape)
        
        plt.plot(ex_test_target_UN[:, stop_num, 0], label="Target")
        plt.plot(out_UN_mean, label="Predictions")
        plt.fill_between(range(len(out_UN_mean)), out_UN_mean - out_UN_std, out_UN_mean + out_UN_std, label="Predictions+", alpha=0.5)
        plt.plot(out_UN_mean - out_UN_std, label="Predictions+")
        plt.plot(out_UN_mean + out_UN_std, label="Predictions+")
        plt.xlabel("Time (3 min intervals)")
        plt.ylabel("Flow (cars/min")
        plt.title("Flow prediction for 15 minutes ahead")
        # plt.fill_between(range(ex_test_target_UN.shape[0]), ex_test_target_UN[:, stop_num, time_step], out_UN[:, stop_num, time_step])
        plt.legend()
        plt.show()
        
        plt.plot(out_UN_std)
        plt.show()