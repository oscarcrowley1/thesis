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
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save, new_generate_dataset

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
    

    A, X, means, stds, info_string = load_scats_data()




    #print_save(f, A)

    # split_line1 = int(X.shape[2] * 0.1)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)


    
    # total_input, total_target = generate_dataset(X,
    #                                                    num_timesteps_input=num_timesteps_input,
    #                                                    num_timesteps_output=num_timesteps_output)

    new_total_input, new_total_target, new_num_timesteps = new_generate_dataset(X)


    print(f"Input comparison:\tNEW:{new_total_input.shape}")
    print(f"Target comparison:\tNEW:{new_total_target.shape}")

    test_indx = np.arange((480*36), (480*45)) # day 36 mon to 44 tues inclusive

    before_indx = np.arange((480*36))
    after_indx = np.arange((480*45), new_total_input.shape[0])

    print(before_indx)
    print(after_indx)

    other_indx = np.concatenate((before_indx, after_indx))
    print(other_indx)
    np.random.shuffle(other_indx)
    split_line = int(other_indx.shape[0] * (5/9))
    training_indx = other_indx[:split_line]
    val_indx = other_indx[split_line:]

    # split_line1 = int(total_input.shape[0] * 0.6)#0.6
    # split_line2 = int(total_input.shape[0] * 0.9)#0.8

    # plt.scatter(training_indx, range(training_indx.shape[0]))
    # plt.scatter(val_indx, range(val_indx.shape[0]))
    # plt.scatter(test_indx, range(test_indx.shape[0]))
    plt.scatter(training_indx, training_indx)
    plt.scatter(val_indx, val_indx)
    plt.scatter(test_indx, test_indx)
    plt.show()

    training_input = new_total_input[training_indx, :, :, :]
    training_target = new_total_target[training_indx, :, :]

    val_input = new_total_input[val_indx, :, :, :]
    val_target = new_total_target[val_indx, :, :]

    test_input = new_total_input[test_indx, :, :, :]
    test_target = new_total_target[test_indx, :, :]

    print("END")

    # train_original_data = X[:, :, training_indx]
    # val_original_data = X[:, :, val_indx]
    # # test_original_data = X[:, :, test_indxsplit_line3]
    # test_original_data = X[:, :, test_indx]

    # training_input, training_target = generate_dataset(train_original_data,
    #                                                    num_timesteps_input=num_timesteps_input,
    #                                                    num_timesteps_output=num_timesteps_output)
    # val_input, val_target = generate_dataset(val_original_data,
    #                                          num_timesteps_input=num_timesteps_input,
    #                                          num_timesteps_output=num_timesteps_output)
    # test_input, test_target = generate_dataset(test_original_data,
    #                                            num_timesteps_input=num_timesteps_input,
    #                                            num_timesteps_output=num_timesteps_output)
