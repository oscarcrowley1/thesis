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
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'DC_STGCN')

from stgcn import STGCN
from utils import generate_dataset, load_scats_data, get_normalized_adj, print_save, generate_feature_vects, get_results, generate_test_feature_vects, generate_test_flow_only_feature_vects, load_test_scats_data

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


    model_int = 11
    
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
    elif model_int == 9:
        """Bravo Junctions. Dropout 0.3. Adjacency values epsilon=1.5, delta squared=10. WITH PRUNING. 4000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_i4000_nodist_2503_e15d10_PRUNING/model_0401_0836_e2999_out1"
        junction_set = "bravo"
    elif model_int == 10:
        """Bravo Plus Junctions. Dropout 0.3. Adjacency values epsilon=1.5, delta squared=10. WITH PRUNING. 2700 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravoplus_d03_nodist_2503_e15d10_PRUNING/model_0408_1153_e2699_out1"
        junction_set = "bravoplus"
    elif model_int == 11:
        """Bravo Junctions. Dropout 0.3. Adjacency values epsilon=1.5, delta squared=10. WITH PRUNING. 1000 iterations. 1 channel input. No distribution"""
        model_string = "final_models/bravo_d03_FLOW_nodist_2503_e15d10_PRUNING/model_0406_1729_e999_out1"
        junction_set = "bravo"
    else:
        print("ERROR NO MODEL CHOSEN")
        
    folder_string = (model_string.rsplit("/", 1))[0]


    A, wrong_X, wrong_means, wrong_stds, wrong_info_string = load_scats_data(folder_string)
    #print(means.shape)


    num_timesteps_input = 25
    num_timesteps_output = 1

    #print_save(f, A)

    # ex_split_line1 = int(X.shape[2] * 0.8)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)
    date_strings = ["jan1_jan14","jan29_feb11","feb5_feb18","jun17_jun30"]
    
    for date_string in date_strings:
        test_string = junction_set + "_" + date_string + "_data"
        X, means, stds, info_string = load_test_scats_data(test_string)
        
        # total_input, total_target, num_timesteps_input = generate_feature_vects(X)
        test_input, test_target, num_timesteps_input = generate_test_flow_only_feature_vects(X)

        # test_input = total_input[ex_split_line1:, :, :]
        # test_target = total_target[ex_split_line1:, :, :]

        A_wave = get_normalized_adj(A)
        A_wave = torch.from_numpy(A_wave)#removed to device

        print(f"XXX{A_wave.shape}")

        ex_net = STGCN(A_wave.shape[0],
                    test_input.shape[3],
                    num_timesteps_input,
                    num_timesteps_output)#.to(device=args.device)
        
        print(f"{A_wave.shape[0]},{test_input.shape[3]},{num_timesteps_input},{num_timesteps_output}")

        
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
                
                plot_time = np.array(range(test_target_UN[:, stop_num, 0].shape[0]))/480
                print(plot_time)
                
                plt.scatter(plot_time, test_target_UN[:, stop_num, 0], alpha=0.5, marker='.', label="Target")
                
                # svr_preds = np.load("svr/" + str(junction_set) + "_station_" + str(stop_num) + ".npy")
                # plt.plot(plot_time, svr_preds, label="SVR Predictions")
                # plt.plot(plot_time, test_target_UN[:, stop_num, 0])
                svr_preds = np.load("svr/" + str(junction_set) + "_station_" + str(stop_num) + "_" + date_string + ".npy")
                plt.plot(plot_time, svr_preds, color='g', label="SVR Predictions")
                plt.plot(plot_time, out_UN[:, stop_num, 0],  color='m', label="STGCN Predictions")
                np.save(folder_string + "/stop" + str(stop_num) + "_preds_test" + test_string, out_UN[:, stop_num, 0])
            
                zero_indexes = np.where(test_target_UN[:, stop_num, 0]==0)
                plt.scatter(plot_time[zero_indexes], (test_target_UN[:, stop_num, 0])[zero_indexes], marker='x', color='r', label="Zero Flow")
                # plt.title(f"Flow for Sensor {stop_num} in Set Bravo Plus")
                plt.xlabel("Time (days)")
                plt.ylabel("Flow (vehicles/hour)")
                plt.title(f"Sensor {stop_num} Flow prediction for 15 minutes ahead")
                # plt.fill_between(range(ex_test_target_UN.shape[0]), ex_test_target_UN[:, stop_num, time_step], out_UN[:, stop_num, time_step])
                plt.legend()
                plt.show()
                
                # abs_svr_error = np.abs(test_target_UN[:, stop_num, 0] - svr_preds)
                # abs_stgcn_error = np.abs(test_target_UN[:, stop_num, 0] - out_UN[:, stop_num, 0])
                # plt.plot(plot_time, abs_svr_error)
                # plt.plot(plot_time, abs_stgcn_error)
                # plt.show()
                
                # plt.plot(plot_time, abs_svr_error - abs_stgcn_error)
                # plt.show()

                
                print(f"\nStop number:\t{stop_num}\tTest Set\t{date_string}")
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
            
            
            df.to_csv(folder_string + "/results_" + test_string + ".csv")
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

            print("\nResults together")
        
            mse, mae, mape, rmse, ev = get_results(test_target_UN[:, :, 0], out_UN[:, :, 0])
            
            may7 = np.arange(0, 480)
            may14 = np.arange(480*7, 480*8)
            may1213 = np.arange(480*5, 480*7)
            weekdays = np.append(np.arange(480, 480*5), np.arange(480*7, 480*9))
            
            # print("\nJust for May 7")
            # mse, mae, mape, rmse, ev = get_results(test_target_UN[may7, :, 0], out_UN[may7, :, 0])
            
            # print("\nJust for May 14")
            # mse, mae, mape, rmse, ev = get_results(test_target_UN[may14, :, 0], out_UN[may14, :, 0])
            
            # print("\nJust for May 12 and 13")
            # mse, mae, mape, rmse, ev = get_results(test_target_UN[may1213, :, 0], out_UN[may1213, :, 0])
            
            # print("\nWeekdays")
            # mse, mae, mape, rmse, ev = get_results(test_target_UN[weekdays, :, 0], out_UN[weekdays, :, 0])

            day_mses, day_maes, day_mapes, day_rmses, day_evs, day_avgs, day_stddevs = [], [], [], [], [], [], []

            num_days = round(test_target_UN.shape[0]/480)-1

            for day in range(num_days):
                print(f"Results for Day {day}")
                day_range = np.arange(480*day, 480*(day+1))
                day_mse, day_mae, day_mape, day_rmse, day_ev = get_results(test_target_UN[day_range, :, 0], out_UN[day_range, :, 0])
                
                day_mses.append(day_mse)
                day_maes.append(day_mae)
                day_mapes.append(day_mape)
                day_rmses.append(day_rmse)
                day_evs.append(day_ev)
                
                all_stops_that_day = test_target_UN[day_range, :, 0]
                
                day_avgs.append(np.mean(np.array(test_target_UN[day_range, :, 0])))
                day_stddevs.append(np.std(np.array(test_target_UN[day_range, :, 0])))

            df = pd.DataFrame()
            
            node_list = list(range(num_days))
            # node_list.append("Average")
            # node_list.append("Standard Deviation")
            df["NODE"] = node_list
            
            df["AVG"] = day_avgs
            df["STDDEV"] = day_stddevs
            
            df["MSE"] = day_mses
            df["MAE"] = day_maes
            df["MAPE"] = day_mapes
            df["RMSE"] = day_rmses
            df["EV"] = day_evs
            
            df.to_csv(folder_string + "/day_results_" + test_string + ".csv")