from cProfile import label
import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from time import process_time
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import sys


from stgcn import STGCN
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save

if __name__ == '__main__':
    A, X, means, stds, info_string = load_scats_data()


    num_timesteps_input = 30
    num_timesteps_output = 15

    #print_save(f, A)

    ex_split_line1 = int(X.shape[2] * 0.8)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)

    
    total_input, total_target = generate_dataset(X,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)

    ex_test_input = total_input[:, :, ex_split_line1:]
    ex_test_target = total_target[:, :, ex_split_line1:]

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)#removed to device

    ex_net = STGCN(A_wave.shape[0],
                ex_test_input.shape[3],
                num_timesteps_input,
                num_timesteps_output)#.to(device=args.device)

    ex_net.load_state_dict(torch.load("saved_models/my_model"))