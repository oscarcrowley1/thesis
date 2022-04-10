from cProfile import label
import os
import argparse
import pickle as pk
from platform import node
from re import S
from tracemalloc import stop
from webbrowser import get
import numpy as np
import matplotlib.pyplot as plt
from sympy import appellf1
import torch
import torch.nn as nn
from time import process_time
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import sys
import pandas as pd


from stgcn import STGCN
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save, generate_feature_vects, get_results

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    print("\nPARAMETERS:")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        # print(f"{name}:\t{parameter}")
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':


    model_int = 2
    if model_int == 0:
        """Alpha Junctions. Dropout 0.3. Adjacency values epsilon=0.6, delta squared=10. 1000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/alpha_d03_nodist_2403/model_0324_1627_e999_out1"
        junction_set = "alpha"
    elif model_int == 1:
        """Bravo Junctions. Dropout 0.3. Adjacency values epsilon=0.6, delta squared=10. 1000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_nodist_2403_e06d10/model_0324_2131_e999_out1"
        junction_set = "bravo"
    elif model_int == 2:
        """Bravo Junctions. Dropout 0.3. Adjacency values epsilon=1.5, delta squared=10. 1000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_nodist_2503_e15d10/model_0325_1957_e999_out1_LATEST"
        junction_set = "bravo"
    else:
        print("ERROR NO MODEL CHOSEN")
        
    folder_string = (model_string.rsplit("/", 1))[0]


    A, X, means, stds, info_string = load_scats_data(folder_string)
    #print(means.shape)


    num_timesteps_input = 25
    num_timesteps_output = 1

    #print_save(f, A)

    ex_split_line1 = int(X.shape[2] * 0.8)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)

    

    
    # total_input, total_target, num_timesteps_input = generate_feature_vects(X)
    training_input, training_target, val_input, val_target, test_input, test_target, num_timesteps_input = generate_feature_vects(X)

    # test_input = total_input[ex_split_line1:, :, :]
    # test_target = total_target[ex_split_line1:, :, :]

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)#removed to device

    # print(f"XXX{A_wave.shape}")

    ex_net = STGCN(A_wave.shape[0],
                test_input.shape[3],
                num_timesteps_input,
                num_timesteps_output)#.to(device=args.device)
    
    ex_net.load_state_dict(torch.load(model_string, map_location=torch.device('cpu')))#for use on my computer

    for name, parameter in ex_net.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print(f"{name}:\t{param}")
        if param == 1024:
            matrix = parameter
            print(type(matrix))
            print(matrix.data.shape)
        # table.add_row([name, param])