from cProfile import label
import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from tables import split_type
import torch
import torch.nn as nn
from time import process_time
#from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


from stgcn import STGCN
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save

#writer = SummaryWriter()


use_gpu = True #CHANGE FOR MY COMPUTER???
num_timesteps_input = 30
num_timesteps_output = 15

plot_rate = 20

# num_timesteps_input = 15
# num_timesteps_output = 15

epochs = 1000
batch_size = 32

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
#if args.enable_cuda and torch.cuda.is_available():
if use_gpu and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')



print(f"Device: {args.device}")


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        print("Batch: {}".format(i), end='\r')
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    print("Batch: {}".format(i))
    print("Finished Loops")
    return sum(epoch_training_losses)/len(epoch_training_losses)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params




if __name__ == '__main__':
    f = open("STGCN-PyTorch-master/run_info.txt", "w")
# info_string = "Epsilon:\t" + str(epsilon) + "\nDelta Squared:\t" + str(delta_squared) + "\nUses these distances\n" + str(coord_array)

# f.close()
    print_save(f, "Begin Setup")
    rand_seed = 7
    print_save(f, f"Random Seed:\t{rand_seed}")
    torch.manual_seed(rand_seed)

    print_save(f, f"Input Timesteps:\t{num_timesteps_input}")
    print_save(f, f"Output Timesteps:\t{num_timesteps_output}")
    print_save(f, f"Use GPU:\t{use_gpu}")
    print_save(f, f"Plot Rate:\t{plot_rate}")
    print_save(f, f"Epochs:\t{epochs}")
    print_save(f, f"Batch Size:\t{batch_size}")

    A, X, means, stds, info_string = load_scats_data()

    print_save(f, info_string)

    #print_save(f, A)

    # split_line1 = int(X.shape[2] * 0.1)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)

    print_save(f, "Split Data")
    
    total_input, total_target = generate_dataset(X,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)

    split_line1 = int(total_input.shape[0] * 0.6)#0.6
    split_line2 = int(total_input.shape[0] * 0.9)#0.8

    training_input = total_input[:split_line1, :, :, :]
    training_target = total_input[:split_line1, :, :]

    val_input = total_input[split_line1:split_line2, :, :, :]
    val_target = total_input[split_line1:split_line2, :, :]

    test_input = total_input[split_line2:, :, :, :]
    test_target = total_input[split_line2:, :, :]

    print("END")

    # train_original_data = X[:, :, :split_line1]
    # val_original_data = X[:, :, split_line1:split_line2]
    # # test_original_data = X[:, :, split_line2:split_line3]
    # test_original_data = X[:, :, split_line2:]

    # training_input, training_target = generate_dataset(train_original_data,
    #                                                    num_timesteps_input=num_timesteps_input,
    #                                                    num_timesteps_output=num_timesteps_output)
    # val_input, val_target = generate_dataset(val_original_data,
    #                                          num_timesteps_input=num_timesteps_input,
    #                                          num_timesteps_output=num_timesteps_output)
    # test_input, test_target = generate_dataset(test_original_data,
    #                                            num_timesteps_input=num_timesteps_input,
    #                                            num_timesteps_output=num_timesteps_output)
