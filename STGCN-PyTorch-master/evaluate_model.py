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
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save, new_generate_dataset, get_results

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


    model_int = 8
    
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
    elif model_int == 3:
        """Bravo Junctions. Dropout 0.3. Adjacency values epsilon=3.2, delta squared=10. 1000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_nodist_2503_e32d10/model_0329_1656_e999_out1"
        junction_set = "bravo"
    elif model_int == 4:
        """Bravo Junctions. Dropout 0.3. Diagonal adjacency matrix. 840 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_nodist_2903_diag/model_0329_2151_e839_out1"
        junction_set = "bravo"
    elif model_int == 5:
        """Bravo Junctions. Dropout 0.3. TRANSPOSED Adjacency values epsilon=1.5, delta squared=10. 880 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_nodist_2903_TRANS_e15d10/model_0330_1147_e879_out1"
        junction_set = "bravo"
    elif model_int == 6:
        """Bravo Junctions. Dropout 0.3. Adjacency values epsilon=1.5, delta squared=10. WITH PRUNING. 1000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_nodist_2503_e15d10_PRUNING/model_0330_1703_e999_out1"
        junction_set = "bravo"
    elif model_int == 7:
        """Bravo Junctions. Dropout 0.3. Adjacency values epsilon=1.5, delta squared=10. MORE FINE TUNED PRUNING. 1000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_nodist_2503_e15d10_MOREPRUNING/model_0331_1616_e999_out1"
        junction_set = "bravo"
    elif model_int == 8:
        """Bravo Junctions. Dropout 0.3. Adjacency values epsilon=0.6, delta squared=10. PRUNING. 1000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/alpha_d03_nodist_2403_PRUNING/model_0331_1958_e999_out1"
        junction_set = "alpha"
        
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

    

    
    # total_input, total_target, num_timesteps_input = new_generate_dataset(X)
    training_input, training_target, val_input, val_target, test_input, test_target, num_timesteps_input = new_generate_dataset(X)

    # test_input = total_input[ex_split_line1:, :, :]
    # test_target = total_target[ex_split_line1:, :, :]

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)#removed to device

    print(f"XXX{A_wave.shape}")

    ex_net = STGCN(A_wave.shape[0],
                test_input.shape[3],
                num_timesteps_input,
                num_timesteps_output)#.to(device=args.device)
    
    total_params = count_parameters(ex_net)
    
    # if False:#if torch.cuda.is_available():
    #     ex_net.load_state_dict(torch.load("saved_models/model_0222_1341_e299"))
    # else:
        # ex_net.load_state_dict(torch.load("saved_models/model_0302_1645_e999", map_location=torch.device('cpu')))#for use on my computer
        # ex_net.load_state_dict(torch.load("saved_models/model_0303_1700_e299", map_location=torch.device('cpu')))#for use on my computer
    ex_net.load_state_dict(torch.load(model_string, map_location=torch.device('cpu')))#for use on my computer
    
    with torch.no_grad():
        ex_net.eval()
    
        out = ex_net(A_wave, test_input)
        # print(ex_test_input.shape)
        # print(ex_test_target.shape)
        # print(out.shape)
        mses = []
        maes = []
        mapes = []
        rmses = []
        evs = []
        avgs = []
        stddevs = []
        
        for stop_num in range(out.shape[1]):
        #stop_num = 7 # 5high 1low
            time_step = 0
        
        
            test_target_UN = test_target*stds[0]+means[0]
            out_UN = out*stds[0]+means[0]
            
            plot_time = np.array(range(test_target_UN[:, stop_num, 0].shape[0]))/20
            print(plot_time)
            
            plt.plot(plot_time, test_target_UN[:, stop_num, 0], label="Target")
            plt.plot(plot_time, out_UN[:, stop_num, 0], label="Predictions")
            print(f"\nStop number:\t{stop_num}")
            mse, mae, mape, rmse, ev = get_results(test_target_UN[:, stop_num, 0], out_UN[:, stop_num, 0])
            
            mses.append(mse)
            maes.append(mae)
            mapes.append(mape)
            rmses.append(rmse)
            evs.append(ev)
            
            print('Average Number of Vehicles: ', np.mean(np.array(test_target_UN[:, stop_num, 0])))
            print('STDDEV of Vehicles: ', np.std(np.array(test_target_UN[:, stop_num, 0])))
            avgs.append(np.mean(np.array(test_target_UN[:, stop_num, 0])))
            stddevs.append(np.std(np.array(test_target_UN[:, stop_num, 0])))
        
            plt.xlabel("Time (hours)")
            plt.ylabel("Flow (cars/interval)")
            plt.title("Sensor {stop_num} Flow prediction for 15 minutes ahead")
            # plt.fill_between(range(ex_test_target_UN.shape[0]), ex_test_target_UN[:, stop_num, time_step], out_UN[:, stop_num, time_step])
            plt.legend()
            plt.show()
            
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
        
        node_list = list(range(out.shape[1]))
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
        
        
        df.to_csv(folder_string + "/results.csv")
        # print(df)
                
        print("\nAverage across all stations")

        print('MSE: ', round((mse_avg),4))
        print('MAE: ', round(mae_avg,4))
        print('MAPE: ', round(mape_avg,4))
        print('RMSE: ', round(rmse_avg,4))
        print('explained_variance: ', round(ev_avg,4))  
        print('Average Number of Vehicles: ', round(avg_avg,4))  
        print('STDDEV of Vehicles: ', round(stddev_avg,4))    
        

        print("\nStandard Deviation across all stations")

        print('MSE: ', round((mse_std),4))
        print('MAE: ', round(mae_std,4))
        print('MAPE: ', round(mape_std,4))
        print('RMSE: ', round(rmse_std,4))
        print('explained_variance: ', round(ev_std,4))  
        print('Average Number of Vehicles: ', round(avg_std,4))  
        print('STDDEV of Vehicles: ', round(stddev_std,4))    

        print("\n results together")
    
        mse, mae, mape, rmse, ev = get_results(test_target_UN[:, :, 0], out_UN[:, :, 0])

    