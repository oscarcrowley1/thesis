from cProfile import label
from datetime import datetime
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
import shutil


from stgcn import STGCN, negLogLik_loss
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save, generate_feature_vects, generate_weekday_feature_vects, generate_flow_only_feature_vects, generate_density_only_feature_vects

writer = SummaryWriter()


use_gpu = False #CHANGE FOR MY COMPUTER???
num_timesteps_input = 26
num_output = 1

plot_rate = 20
save_rate = 50

# num_timesteps_input = 15
# num_timesteps_output = 15

epochs = 1500
batch_size = 32
dist_bool = False
data_zip = "bravoplus" # alpha bravo bravoplus
# all_days = True
feature_vec_type = 1 # 0 all days. 1 weekdays only. 2 all days flow only. 3 all days density only
load_model = False

if not dist_bool:
    num_output = 1
else:
    num_output = 2

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

    if dist_bool:
        epoch_training_log_losses = []
        
    epoch_training_classic_losses = []

    for i in range(0, training_input.shape[0], batch_size):
        print("Batch: {}".format(i), end='\r')
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        #print(f"OUTTYPE:\t{out.type}")
        #loss = loss_criterion(out, y_batch)
        # print(out[..., 0].shape)
        # print(y_batch.shape)
        
        
        if dist_bool:
            log_loss = negLogLik_loss(out, y_batch)
            classic_loss = loss_criterion(out[:, :, 0][..., None], y_batch)
            epoch_training_log_losses.append(log_loss.detach().cpu().numpy())
            log_loss.backward()
        else:
            classic_loss = loss_criterion(out, y_batch)
            classic_loss.backward()
            
        epoch_training_classic_losses.append(classic_loss.detach().cpu().numpy())
            
        optimizer.step()
    print("Batch: {}".format(i))
    print("Finished Loops")
    if dist_bool:
        average_loss_s = sum(epoch_training_log_losses)/len(epoch_training_log_losses), sum(epoch_training_classic_losses)/len(epoch_training_classic_losses)
    else:
        average_loss_s = sum(epoch_training_classic_losses)/len(epoch_training_classic_losses)

    return average_loss_s

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    print_save(f, "\nPARAMETERS:")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #print_save(f, f"{name}:\t{parameter}")
        table.add_row([name, param])
        total_params+=param
    print(table)
    print_save(f, f"Total Trainable Params: {total_params}")
    return total_params



if __name__ == '__main__':
    f = open("runs/run_info.txt", "w")
# info_string = "Epsilon:\t" + str(epsilon) + "\nDelta Squared:\t" + str(delta_squared) + "\nUses these distances\n" + str(coord_array)

# f.close()
    print_save(f, "Begin Setup")
    rand_seed = 8 # was 7
    print_save(f, f"Random Seed:\t{rand_seed}")
    torch.manual_seed(rand_seed)

    print_save(f, f"Input Timesteps:\t{num_timesteps_input}")
    print_save(f, f"Output Timesteps:\t{num_output}")
    print_save(f, f"Use GPU:\t{use_gpu}")
    print_save(f, f"Plot Rate:\t{plot_rate}")
    print_save(f, f"Epochs:\t{epochs}")
    print_save(f, f"Batch Size:\t{batch_size}")
    print_save(f, f"Distribution:\t{dist_bool}")
    print_save(f, f"Load Model:\t{load_model}")
    print_save(f, f"Feature Vector Type:\t{feature_vec_type}")

    A, X, means, stds, info_string = load_scats_data(data_zip)

    print_save(f, info_string)

    #print_save(f, A)

    # split_line1 = int(X.shape[2] * 0.1)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)

    print_save(f, "Split Data")
    
    # total_input, total_target, num_timesteps_input = generate_feature_vects(X)

    if feature_vec_type == 0:
        training_input, training_target, val_input, val_target, test_input, test_target, num_timesteps_input = generate_feature_vects(X)
    elif feature_vec_type == 1:
        training_input, training_target, val_input, val_target, test_input, test_target, num_timesteps_input = generate_weekday_feature_vects(X)
    elif feature_vec_type == 2:
        training_input, training_target, val_input, val_target, test_input, test_target, num_timesteps_input = generate_flow_only_feature_vects(X)
    elif feature_vec_type == 3:
        training_input, training_target, val_input, val_target, test_input, test_target, num_timesteps_input = generate_density_only_feature_vects(X)

    print_save(f, "Shuffle Data")

    # test_indx = np.arange((480*36), (480*45)) # day 36 mon to 44 tues inclusive

    # before_indx = np.arange((480*36))
    # after_indx = np.arange((480*45), total_input.shape[0])

    # print(before_indx)
    # print(after_indx)

    # other_indx = np.concatenate((before_indx, after_indx))
    # print(other_indx)
    # np.random.shuffle(other_indx)
    # split_line = int(other_indx.shape[0] * (5/9))
    # training_indx = other_indx[:split_line]
    # val_indx = other_indx[split_line:]

    # test_indx = np.arange((480*36), (480*45)) # day 36 mon to 44 tues inclusive

    # other_indx = np.concatenate(np.arange((480*36)), np.arange((480*45), total_input.shape[0]))
    # np.random.shuffle(other_indx)
    # split_line = int(other_indx.shape * (5/9))
    # training_indx = other_indx[:split_line]
    # val_indx = other_indx[split_line:]

    # rand_indx = torch.randperm(total_input.shape[0])

    # split_line1 = int(total_input.shape[0] * 0.6)#0.6
    # split_line2 = int(total_input.shape[0] * 0.9)#0.8

    # split_line1 = int(total_input.shape[0] * 0.5)#0.6
    # split_line2 = int(total_input.shape[0] * 0.9)#0.8

    # training_indx = rand_indx[:split_line1]
    # val_indx = rand_indx[split_line1:split_line2]
    # test_indx = rand_indx[split_line2:]

    # training_input = total_input[:split_line1, :, :, :]
    # training_target = total_target[:split_line1, :, :]

    # val_input = total_input[split_line1:split_line2, :, :, :]
    # val_target = total_target[split_line1:split_line2, :, :]

    # test_input = total_input[split_line2:, :, :, :]
    # test_target = total_target[split_line2:, :, :]

    # training_input = total_input[training_indx, :, :, :]
    # training_target = total_target[training_indx, :, :]

    # val_input = total_input[val_indx, :, :, :]
    # val_target = total_target[val_indx, :, :]

    # test_input = total_input[test_indx, :, :, :]
    # test_target = total_target[test_indx, :, :]

    # split_line1 = int(X.shape[2] * 0.6)#0.6
    # split_line2 = int(X.shape[2] * 0.9)#0.8

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

    # print(f"SHapes: {training_input.shape}, {training_input_test.shape}")

    print_save(f, "Normalise Adjacency Matrix")

    A_wave = get_normalized_adj(A)

    #print_save(f, A_wave)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    print_save(f, "Initialise STGCN")

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_output).to(device=args.device)


    if load_model == True:
        model_string = "final_models/bravo_d03_nodist_2503_e15d10_PRUNING/model_0330_1703_e999_out1"
        net.load_state_dict(torch.load(model_string))

    # writer.add_graph(net, (A_wave, val_input))
    
    # writer.flush()
    # writer.close()
    # sys.exit()
    
    print_save(f, str(count_parameters(net)))

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()
    #loss_criterion = negLogLik_loss()

    if dist_bool:
        training_log_losses = []
        validation_log_losses = []

    training_classic_losses = []
    validation_classic_losses = []

    validation_maes = []
    validation_mses = []

    #output_array = []

    print_save(f, "Begin Training")

    f.close()
    
    training_start = process_time()
    x_epoch_start = process_time()

    for epoch in range(epochs):
        epoch_start = process_time()
        print("Epoch Number: {}".format(epoch))
        print("Epoch Number: {}".format(epoch))
        if dist_bool:
            log_loss, classic_loss = train_epoch(training_input, training_target,
                            batch_size=batch_size)
        else:
            classic_loss = train_epoch(training_input, training_target,
                            batch_size=batch_size)
        print("Returned Losses")
        if dist_bool:
            training_log_losses.append(log_loss)
            writer.add_scalar("Training Log Loss", log_loss, epoch)
        
        training_classic_losses.append(classic_loss)
        writer.add_scalar("Training Classic Loss", classic_loss, epoch)
        
        # Run validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)

            out = net(A_wave, val_input)
            #val_loss = loss_criterion(out, val_target).to(device="cpu")
            if dist_bool:
                val_log_loss = negLogLik_loss(out, val_target).to(device="cpu")
                validation_log_losses.append((val_log_loss.detach().numpy()).item())
                val_classic_loss = loss_criterion(out[:, :, 0][..., None], val_target).to(device="cpu")
            else:
                val_classic_loss = loss_criterion(out, val_target).to(device="cpu")
            #validation_losses.append(np.asscalar(val_loss.detach().numpy()))
            validation_classic_losses.append((val_classic_loss.detach().numpy()).item())

            # out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
            # target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]


            # # if (epoch+1)%plot_rate==0:
            # #     plt.plot(out_unnormalized[:, 0, 2], label="Out")
            # #     plt.plot(target_unnormalized[:, 0, 2], label="Target")
            # #     plt.legend()
            # #     plt.show()

            # mae = np.mean(np.absolute(out_unnormalized - target_unnormalized)) #why would mae be calculated after normalisation
            # rmse = np.sqrt(np.mean((out_unnormalized - target_unnormalized)**2))

            #mae_15min = np.mean(np.absolute(out_unnormalized[:,:,4] - target_unnormalized[:,:,4]))

            # validation_maes.append(mae) 3854, 281373

            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")
        
        if dist_bool:
            writer.add_scalar("Validation Log Loss", val_log_loss, epoch)
        
        writer.add_scalar("Validation Classic Loss", val_classic_loss, epoch)
        # writer.add_scalar("Validation MAE", mae, epoch)
        # #writer.add_scalar("Validation MAE 15th min", rmse, epoch)
        # writer.add_scalar("Validation MSE Unnormalised", rmse, epoch)
        if dist_bool:
            print("Training loss: {}".format(training_log_losses[-1]))
            print("Validation loss: {}".format(validation_log_losses[-1]))
        else:
            print("Training loss: {}".format(training_classic_losses[-1]))
            print("Validation loss: {}".format(validation_classic_losses[-1]))
        #print("Validation MAE: {}".format(validation_maes[-1]))
        #print(f"THE LENGTHS: {training_losses}\t{validation_losses}\t{validation_maes}")
        # if (epoch+1)%plot_rate==0:
        #     x_epoch_end = process_time()
        #     print(f"Time for {plot_rate} epochs:\t{x_epoch_end-x_epoch_start}")
        #     x_epoch_start = x_epoch_end
        #     plt.plot(training_losses, label="training loss")
        #     plt.plot(validation_losses, label="validation loss")
        #     plt.legend()
        #     plt.show()
        #     print("PLOTTED")
        if (epoch+1) % save_rate == 0:
            now = datetime.now()
            time_string = now.strftime("%m%d_%H%M") + "_e" + str(epoch) + "_out" + str(num_output)

            torch.save(net.state_dict(), ("saved_models/model_" + time_string))
            shutil.copy("runs/run_info.txt", "saved_models/run_info_" + time_string)

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            if dist_bool:
                pk.dump((training_log_losses, validation_log_losses, validation_maes), fd)
            else:
                pk.dump((training_classic_losses, validation_classic_losses, validation_maes), fd)

        epoch_stop = process_time()
        epoch_length = epoch_stop-epoch_start
        print(f"Epoch Time:\t{epoch_length}")
        writer.add_scalar("Epoch Length", epoch_length, epoch)
            
    writer.flush()
    writer.close()
    
    now = datetime.now()
    time_string = now.strftime("%m%d_%H%M") + "_e" + str(epoch) + "_out" + str(num_output)

    torch.save(net.state_dict(), ("saved_models/model_" + time_string))

    training_stop = process_time()
    print(f"Training Time:\t{training_stop-training_start}")