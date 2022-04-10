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
import pandas as pd
from joblib import dump, load

# Allows us to use files form this folder
sys.path.insert(1, 'DC_STGCN/')
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save, generate_feature_vects, get_results

# writer = SummaryWriter()


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


# def train_epoch(training_input, training_target, batch_size):
#     """
#     Trains one epoch with the given data.
#     :param training_input: Training inputs of shape (num_samples, num_nodes,
#     num_timesteps_train, num_features).
#     :param training_target: Training targets of shape (num_samples, num_nodes,
#     num_timesteps_predict).
#     :param batch_size: Batch size to use during training.
#     :return: Average loss for this epoch.
#     """
#     permutation = torch.randperm(training_input.shape[0])

#     epoch_training_losses = []
#     for i in range(0, training_input.shape[0], batch_size):
#         print("Batch: {}".format(i), end='\r')
#         net.train()
#         optimizer.zero_grad()

#         indices = permutation[i:i + batch_size]
#         X_batch, y_batch = training_input[indices], training_target[indices]
#         X_batch = X_batch.to(device=args.device)
#         y_batch = y_batch.to(device=args.device)

#         out = net(A_wave, X_batch)
#         loss = loss_criterion(out, y_batch)
#         loss.backward()
#         optimizer.step()
#         epoch_training_losses.append(loss.detach().cpu().numpy())
#     print("Batch: {}".format(i))
#     print("Finished Loops")
#     return sum(epoch_training_losses)/len(epoch_training_losses)


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
#     f = open("DC_STGCN/run_info.txt", "w")
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
    data_string = "bravoplus"

    A, X, means, stds, info_string = load_scats_data(data_string)

    # print_save(f, info_string)

    #print_save(f, A)

    # split_line1 = int(X.shape[2] * 0.1)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)

    # print_save(f, "Split Data")
    
    # total_input, total_target, num_timesteps_input = generate_feature_vects(X)
    training_input, training_target, val_input, val_target, test_input, test_target, num_features = generate_feature_vects(X)
    # print_save(f, "Shuffle Data")

    # rand_indx = torch.randperm(total_input.shape[0])

    # # split_line1 = int(total_input.shape[0] * 0.6)#0.6
    # # split_line2 = int(total_input.shape[0] * 0.9)#0.8

    # split_line1 = int(total_input.shape[0] * 0.5)#0.6
    # split_line2 = int(total_input.shape[0] * 0.9)#0.8

    # training_indx = rand_indx[:split_line1]
    # val_indx = rand_indx[split_line1:split_line2]
    # test_indx = rand_indx[split_line2:]

    # # training_input = total_input[:split_line1, :, :, :]
    # # training_target = total_target[:split_line1, :, :]

    # # val_input = total_input[split_line1:split_line2, :, :, :]
    # # val_target = total_target[split_line1:split_line2, :, :]

    # # test_input = total_input[split_line2:, :, :, :]
    # # test_target = total_target[split_line2:, :, :]

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
    mapes = []
    rmses = []
    evs = []
    avgs = []
    stddevs = []
    
    station_day_val = []
    station_dayS_val = []
    all_station_days_val = []

    #output_array = []

    # print_save(f, "Begin Training")

    # f.close()
    
    training_start = process_time()
    x_epoch_start = process_time()
    
    print(f"XSHAPE:\t{X.shape}")
    print(f"Training Input SHAPE:\t{training_input.shape}")
    print(f"Training Target SHAPE:\t{training_target.shape}")
    
    #station_num = 1
    may7 = np.arange(0, 480)
    may14 = np.arange(480*7, 480*8)
    may1213 = np.arange(480*5, 480*7)
    weekdays = np.append(np.arange(480, 480*5), np.arange(480*7, 480*9))
    
    day_nums = np.arange(0,9)
    
    an_day_mses, an_day_maes, an_day_mapes, an_day_rmses, an_day_evs, an_day_avgs, an_day_stddevs = [], [], [], [], [], [], []

    # subset = []
    subset = []
    for station_num in range(training_input.shape[1]):
        print(training_input)
        stationX_training_input = training_input[:, station_num, :, 0]
        stationX_training_target = training_target[:, station_num, 0]
        
        
        
        
        print(f"Training Input SHAPE:\t{training_input.shape}")
        print(f"Training Target SHAPE:\t{training_target.shape}")
        
        model = LinearSVR()
        
        model.fit(stationX_training_input, stationX_training_target)
        
        dump(model, "svr/model_" + str(data_string) + "_" + str(station_num) + ".joblib")
        
        print(f"PARAMS:\t{len(model.get_params())}")
        print(f"COEFS:\t{len(model.coef_)}")
        
        if len(subset) != 0:
            # stationX_training_input = training_input[subset, station_num, :, 0]
            # stationX_training_target = training_target[subset, station_num, 0]
            
            stationX_test_input = test_input[subset, station_num, :, 0]
            stationX_test_target = test_target[subset, station_num, 0]
        else:
            
            stationX_test_input = test_input[:, station_num, :, 0]
            stationX_test_target = test_target[:, station_num, 0]
        
        stationX_test_pred = model.predict(stationX_test_input)
        
        stationX_test_pred = stationX_test_pred*stds[0]+means[0]
        stationX_test_target = stationX_test_target*stds[0]+means[0]
        
        np.save("svr/" + str(data_string) + "_station_" + str(station_num), stationX_test_pred)
        
        print(f"Station:\t{station_num}")
        mse, mae, mape, rmse, ev = get_results(stationX_test_target, stationX_test_pred)
        
        day_mses, day_maes, day_mapes, day_rmses, day_evs, day_avgs, day_stddevs = [], [], [], [], [], [], []

        for i in range(9):
            
            day_mse, day_mae, day_mape, day_rmse, day_ev = get_results(stationX_test_target[i*480:(i+1)*480], stationX_test_pred[i*480:(i+1)*480])
            
            day_mses.append(day_mse)
            day_maes.append(day_mae)
            day_mapes.append(day_mape)
            day_rmses.append(day_rmse)
            day_evs.append(day_ev)
            
        an_day_mses.append(day_mses)
        an_day_maes.append(day_maes)
        an_day_mapes.append(day_mapes)
        an_day_rmses.append(day_rmses)
        an_day_evs.append(day_evs)
        
        mses.append(mse)
        maes.append(mae)
        mapes.append(mape)
        rmses.append(rmse)
        evs.append(ev)
        
        print('Average Number of Vehicles: ', np.mean(np.array(stationX_test_target)))
        print('STDDEV of Vehicles: ', np.std(np.array(stationX_test_target)))
        avgs.append(np.mean(np.array(stationX_test_target)))
        stddevs.append(np.std(np.array(stationX_test_target)))
        
    an_day_mses = np.array(an_day_mses)
    an_day_maes = np.array(an_day_maes)
    an_day_mapes = np.array(an_day_mapes)
    an_day_rmses = np.array(an_day_rmses)
    an_day_evs = np.array(an_day_evs)
    
    avg_day_mses = np.mean(an_day_mses, axis=0)
    avg_day_maes = np.mean(an_day_maes, axis=0)
    avg_day_mapes = np.mean(an_day_mapes, axis=0)
    avg_day_rmses = np.mean(an_day_rmses, axis=0)
    avg_day_evs = np.mean(an_day_evs, axis=0)
    
    print(f"MSES for each DAY:\t{avg_day_mses}")
    print(f"MAES for each DAY:\t{avg_day_maes}")
    print(f"MAPES for each DAY:\t{avg_day_mapes}")
    print(f"RMSES for each DAY:\t{avg_day_rmses}")
    print(f"EVS for each DAY:\t{avg_day_evs}")
        
    print("All stations finished")
    
    print(f"SHAPES:\t{mses}\t{maes}\t{rmses}\t{evs}")

        
    mse_avg = np.mean(np.array(mses))
    mae_avg = np.mean(np.array(maes))
    mape_avg = np.mean(np.array(mapes))
    rmse_avg = np.mean(np.array(rmses))
    ev_avg = np.mean(np.array(evs))
    avg_avg = np.mean(np.array(avgs))
    stddev_avg = np.mean(np.array(stddevs))
    
    mse_std = np.std(np.array(mses))
    mae_std = np.std(np.array(maes))
    mape_std = np.std(np.array(mapes))
    rmse_std = np.std(np.array(rmses))
    ev_std = np.std(np.array(evs))
    avg_std = np.std(np.array(avgs))
    stddev_std = np.std(np.array(stddevs))
    
    mses.append(mse_avg)
    maes.append(mae_avg)
    mapes.append(mape_avg)
    rmses.append(rmse_avg)
    evs.append(ev_avg)
    avgs.append(avg_avg)
    stddevs.append(stddev_avg)
    
    mses.append(mse_std)
    maes.append(mae_std)
    mapes.append(mape_std)
    rmses.append(rmse_std)
    evs.append(ev_std)
    avgs.append(avg_std)
    stddevs.append(stddev_std)
    
    df = pd.DataFrame()
        
    node_list = list(range(training_input.shape[1]))
    node_list.append("Average")
    node_list.append("Standard Deviation")
    df["NODE"] = node_list
    
    df["AVG"] = avgs
    df["STDDEV"] = stddevs
    
    df["MSE"] = mses
    df["MAE"] = maes
    df["MAPE"] = mapes
    df["RMSE"] = rmses
    df["EV"] = evs
    
    # FOR SAVING
    # df.to_csv("svr/results_" + data_string)
    
    print("Average across all stations")

    print('MSE: ', round((mse_avg),4))
    print('MAE: ', round(mae_avg,4))
    print('MAPE: ', round(mape_avg,4))
    print('RMSE: ', round(rmse_avg,4))
    print('explained_variance: ', round(ev_avg,4))  
    print('Average Number of Vehicles: ', round(avg_avg,4))  
    print('STDDEV of Vehicles: ', round(stddev_avg,4))    
    

    print("Standard Deviation across all stations")

    print('MSE: ', round((mse_std),4))
    print('MAE: ', round(mae_std,4))
    print('MAPE: ', round(mape_std,4))
    print('RMSE: ', round(rmse_std,4))
    print('explained_variance: ', round(ev_std,4))  
    print('Average Number of Vehicles: ', round(avg_std,4))  
    print('STDDEV of Vehicles: ', round(stddev_std,4))     
