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
from sklearn.svm import LinearSVR

from stgcn import STGCN
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save, new_generate_dataset, get_results

writer = SummaryWriter()


use_gpu = True #CHANGE FOR MY COMPUTER???
num_timesteps_input = 26
num_timesteps_output = 1

plot_rate = 20
save_rate = 100

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
#     f = open("STGCN-PyTorch-master/run_info.txt", "w")
# # info_string = "Epsilon:\t" + str(epsilon) + "\nDelta Squared:\t" + str(delta_squared) + "\nUses these distances\n" + str(coord_array)

# # f.close()
#     print_save(f, "Begin Setup")
#     rand_seed = 8#was7
#     print_save(f, f"Random Seed:\t{rand_seed}")
#     torch.manual_seed(rand_seed)

#     print_save(f, f"Input Timesteps:\t{num_timesteps_input}")
#     print_save(f, f"Output Timesteps:\t{num_timesteps_output}")
#     print_save(f, f"Use GPU:\t{use_gpu}")
#     print_save(f, f"Plot Rate:\t{plot_rate}")
#     print_save(f, f"Epochs:\t{epochs}")
#     print_save(f, f"Batch Size:\t{batch_size}")

    A, X, means, stds, info_string = load_scats_data()

    # print_save(f, info_string)

    #print_save(f, A)

    # split_line1 = int(X.shape[2] * 0.1)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)

    # print_save(f, "Split Data")
    
    total_input, total_target, num_timesteps_input = new_generate_dataset(X)

    # print_save(f, "Shuffle Data")

    rand_indx = torch.randperm(total_input.shape[0])

    # split_line1 = int(total_input.shape[0] * 0.6)#0.6
    # split_line2 = int(total_input.shape[0] * 0.9)#0.8

    split_line1 = int(total_input.shape[0] * 0.5)#0.6
    split_line2 = int(total_input.shape[0] * 0.9)#0.8

    training_indx = rand_indx[:split_line1]
    val_indx = rand_indx[split_line1:split_line2]
    test_indx = rand_indx[split_line2:]

    # training_input = total_input[:split_line1, :, :, :]
    # training_target = total_target[:split_line1, :, :]

    # val_input = total_input[split_line1:split_line2, :, :, :]
    # val_target = total_target[split_line1:split_line2, :, :]

    # test_input = total_input[split_line2:, :, :, :]
    # test_target = total_target[split_line2:, :, :]

    training_input = total_input[training_indx, :, :, :]
    training_target = total_target[training_indx, :, :]

    val_input = total_input[val_indx, :, :, :]
    val_target = total_target[val_indx, :, :]

    test_input = total_input[test_indx, :, :, :]
    test_target = total_target[test_indx, :, :]

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

    # print_save(f, "Normalise Adjacency Matrix")

    # A_wave = get_normalized_adj(A)

    # #print_save(f, A_wave)
    # A_wave = torch.from_numpy(A_wave)

    # A_wave = A_wave.to(device=args.device)

    # print_save(f, "Initialise STGCN")

    # net = STGCN(A_wave.shape[0],
    #             training_input.shape[3],
    #             num_timesteps_input,
    #             num_timesteps_output).to(device=args.device)
    
    # writer.add_graph(net, (A_wave, val_input))
    
    # writer.flush()
    # writer.close()
    # sys.exit()
    
    # print_save(f, str(count_parameters(net)))

    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []

    validation_maes = []
    validation_mses = []
    
    mses = []
    maes = []
    rmses = []
    evs = []

    #output_array = []

    # print_save(f, "Begin Training")

    # f.close()
    
    training_start = process_time()
    x_epoch_start = process_time()
    
    print(f"XSHAPE:\t{X.shape}")
    print(f"Training Input SHAPE:\t{training_input.shape}")
    print(f"Training Target SHAPE:\t{training_target.shape}")
    
    #station_num = 1
    for station_num in range(9):
        stationX_training_input = training_input[:, station_num, :, 0]
        stationX_training_target = training_target[:, station_num, 0]
        
        stationX_val_input = val_input[:, station_num, :, 0]
        stationX_val_target = val_target[:, station_num, 0]
        
        print(f"Training Input SHAPE:\t{training_input.shape}")
        print(f"Training Target SHAPE:\t{training_target.shape}")
        
        model = LinearSVR()
        
        model.fit(stationX_training_input, stationX_training_target)
        
        print(f"PARAMS:\t{len(model.get_params())}")
        print(f"COEFS:\t{len(model.coef_)}")
        
        stationX_val_pred = model.predict(stationX_val_input)
        
        stationX_val_pred = stationX_val_pred*stds[0]+means[0]
        stationX_val_target = stationX_val_target*stds[0]+means[0]
        
        print(f"Station:\t{station_num}")
        mse, mae, rmse, ev = get_results(stationX_val_target, stationX_val_pred)
        
        mses.append(mse)
        maes.append(mae)
        rmses.append(rmse)
        evs.append(ev)
        
    print("All stations finished")
    
    print(f"SHAPES:\t{mses}\t{maes}\t{rmses}\t{evs}")

        
    mse_avg = np.mean(np.array(mses))
    mae_avg = np.mean(np.array(maes))
    rmse_avg = np.mean(np.array(rmses))
    ev_avg = np.mean(np.array(evs))
    
    print("Average across all stations")

    print('MSE: ', round((mse_avg),4))
    print('MAE: ', round(mae_avg,4))
    print('RMSE: ', round(rmse_avg,4))
    print('explained_variance: ', round(ev_avg,4))    

    
    # stationX_training_input = training_input[:, :, :, 0]
    # stationX_training_target = training_target[:, :, 0]
    
    # stationX_val_input = val_input[:, :, :, 0]
    # stationX_val_target = val_target[:, :, 0]
    
    # print(f"Training Input SHAPE:\t{training_input.shape}")
    # print(f"Training Target SHAPE:\t{training_target.shape}")
    
    # model = LinearSVR()
    
    # model.fit(stationX_training_input, stationX_training_target)
    
    # stationX_val_pred = model.predict(stationX_val_input)
    
    # stationX_val_pred = stationX_val_pred*stds[0]+means[0]
    # stationX_val_target = stationX_val_target*stds[0]+means[0]
    
    # mses.append(get_results(stationX_val_target, stationX_val_pred))
    
    

    # for epoch in range(epochs):
    #     epoch_start = process_time()
    #     print("Epoch Number: {}".format(epoch))
    #     print("Epoch Number: {}".format(epoch))
    #     loss = train_epoch(training_input, training_target,
    #                        batch_size=batch_size)
    #     print("Returned Losses")
    #     training_losses.append(loss)
        
    #     writer.add_scalar("Training Loss", loss, epoch)
        
    #     # Run validation
    #     with torch.no_grad():
    #         net.eval()
    #         val_input = val_input.to(device=args.device)
    #         val_target = val_target.to(device=args.device)

    #         out = net(A_wave, val_input)
    #         val_loss = loss_criterion(out, val_target).to(device="cpu")
    #         #validation_losses.append(np.asscalar(val_loss.detach().numpy()))
    #         validation_losses.append((val_loss.detach().numpy()).item())

    #         out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
    #         target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]


    #         # if (epoch+1)%plot_rate==0:
    #         #     plt.plot(out_unnormalized[:, 0, 2], label="Out")
    #         #     plt.plot(target_unnormalized[:, 0, 2], label="Target")
    #         #     plt.legend()
    #         #     plt.show()

    #         mae = np.mean(np.absolute(out_unnormalized - target_unnormalized)) #why would mae be calculated after normalisation
    #         rmse = np.sqrt(np.mean((out_unnormalized - target_unnormalized)**2))

    #         #mae_15min = np.mean(np.absolute(out_unnormalized[:,:,4] - target_unnormalized[:,:,4]))

    #         validation_maes.append(mae)

    #         out = None
    #         val_input = val_input.to(device="cpu")
    #         val_target = val_target.to(device="cpu")
            
    #     writer.add_scalar("Validation Loss", val_loss, epoch)
    #     writer.add_scalar("Validation MAE", mae, epoch)
    #     #writer.add_scalar("Validation MAE 15th min", rmse, epoch)
    #     writer.add_scalar("Validation MSE Unnormalised", rmse, epoch)

    #     print("Training loss: {}".format(training_losses[-1]))
    #     print("Validation loss: {}".format(validation_losses[-1]))
    #     print("Validation MAE: {}".format(validation_maes[-1]))
    #     #print(f"THE LENGTHS: {training_losses}\t{validation_losses}\t{validation_maes}")
    #     # if (epoch+1)%plot_rate==0:
    #     #     x_epoch_end = process_time()
    #     #     print(f"Time for {plot_rate} epochs:\t{x_epoch_end-x_epoch_start}")
    #     #     x_epoch_start = x_epoch_end
    #     #     plt.plot(training_losses, label="training loss")
    #     #     plt.plot(validation_losses, label="validation loss")
    #     #     plt.legend()
    #     #     plt.show()
    #     #     print("PLOTTED")
    #     if (epoch+1) % save_rate == 0:
    #         now = datetime.now()
    #         time_string = now.strftime("%m%d_%H%M") + "_e" + str(epoch)

    #         torch.save(net.state_dict(), ("saved_models/model_" + time_string))
    #         shutil.copy("STGCN-PyTorch-master/run_info.txt", "run_info_" + time_string)

    #     checkpoint_path = "checkpoints/"
    #     if not os.path.exists(checkpoint_path):
    #         os.makedirs(checkpoint_path)
    #     with open("checkpoints/losses.pk", "wb") as fd:
    #         pk.dump((training_losses, validation_losses, validation_maes), fd)

    #     epoch_stop = process_time()
    #     epoch_length = epoch_stop-epoch_start
    #     print(f"Epoch Time:\t{epoch_length}")
    #     writer.add_scalar("Epoch Length", epoch_length, epoch)
            
    # writer.flush()
    # writer.close()
    
    # torch.save(net.state_dict(), "saved_models/my_model")

    # training_stop = process_time()
    # print(f"Training Time:\t{training_stop-training_start}")