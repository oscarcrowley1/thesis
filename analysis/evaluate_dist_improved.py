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
import datetime
import matplotlib.dates as mdates

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'DC_STGCN')

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


    model_int = 0
    
    if model_int == 0:
        """Alpha Junctions. Dropout 0.3. Adjacency values epsilon=0.6, delta squared=10. 1000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_DIST_2503_e15d10_PRUNING_longrun/model_0406_1258_e4999_out2"
        junction_set = "bravo"
    
    elif model_int == 1:
        """Alpha Junctions. Dropout 0.3. Adjacency values epsilon=0.6, delta squared=10. 1000 iterations. 2 channel input. No distribution"""
        model_string = "final_models/bravo_d03_DIST_2503_e15d10_PRUNING_longrun/model_0406_1258_e4999_out2"
        junction_set = "bravo"
        
    else:
        print("ERROR NO MODEL CHOSEN")
        
    folder_string = (model_string.rsplit("/", 1))[0]


    A, X, means, stds, info_string = load_scats_data(folder_string)
    #print(means.shape)


    num_timesteps_input = 25
    num_timesteps_output = 2

    #print_save(f, A)

    # ex_split_line1 = int(X.shape[2] * 0.8)#0.6
    # split_line2 = int(X.shape[2] * 0.15)#0.8
    # split_line3 = int(X.shape[2] * 0.2)

    

    
    # total_input, total_target, num_timesteps_input = generate_feature_vects(X)
    training_input, training_target, val_input, val_target, test_input, test_target, num_timesteps_input = generate_feature_vects(X)

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
        frac_ins1 = []
        frac_ins2 = []
        frac_ins3 = []
        
        
        start_datetime = datetime.datetime.fromisoformat('2018-05-14')
        interval_length = datetime.timedelta(minutes=3)
        plot_date = [start_datetime+i*interval_length for i in range(test_target.shape[0])]
        print(f"PLOTDATE:\t{plot_date}")
        print(f"PLOTDATE Length:\t{len(plot_date)}\t{test_target.shape[0]}")
        
        test_target_UN = test_target*stds[0]+means[0]
        out_UN_mean = out[:, :, 0]*stds[0]+means[0]
        out_UN_std = out[:, :, 1]*stds[0]
        
        for stop_num in range(out.shape[1]):
        #stop_num = 7 # 5high 1low
            time_step = 0
        
            # print(f"TTSHAPE:\t{test_target_UN.shape}")
            
            test_target_UN_stopnum = test_target_UN[:, stop_num, 0]
            out_UN_mean_stopnum = out_UN_mean[:, stop_num]
            out_UN_mean_stopnum_plot = out_UN_mean_stopnum.clip(min=0)
            out_UN_std_stopnum = out_UN_std[:, stop_num]
            
            plot_time = np.array(range(test_target_UN_stopnum.shape[0]))/480
            print(plot_time)
            # plot_date = [datetime.datetime.fromordinal(time) for time in plot_time]
            # print(plot_date)
            
            greater1 = np.greater_equal(test_target_UN_stopnum, out_UN_mean_stopnum-out_UN_std_stopnum)
            lesser1 = np.less_equal(test_target_UN_stopnum, out_UN_mean_stopnum+out_UN_std_stopnum)
            inside1 = greater1 & lesser1
            frac_in1 = np.count_nonzero(inside1)/len(inside1)
            
            greater2 = np.greater_equal(test_target_UN_stopnum, out_UN_mean_stopnum-2*out_UN_std_stopnum)
            lesser2 = np.less_equal(test_target_UN_stopnum, out_UN_mean_stopnum+2*out_UN_std_stopnum)
            inside2 = greater2 & lesser2
            frac_in2 = np.count_nonzero(inside2)/len(inside2)
            
            greater3 = np.greater_equal(test_target_UN_stopnum, out_UN_mean_stopnum-3*out_UN_std_stopnum)
            lesser3 = np.less_equal(test_target_UN_stopnum, out_UN_mean_stopnum+3*out_UN_std_stopnum)
            inside3 = greater3 & lesser3
            frac_in3 = np.count_nonzero(inside3)/len(inside3)
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M %d/%m/%Y'))


            plt.scatter(plot_date, (test_target_UN_stopnum), alpha=0.5, label="Target", marker='.')
            
            nodist_preds = np.load("final_models/bravo_d03_i4000_nodist_2503_e15d10_PRUNING/stop" + str(stop_num) + "_preds.npy")
            # plt.plot(plot_date, nodist_preds, label="No Dist Predictions")

            # svr_preds = np.load("svr/" + str(junction_set) + "_station_" + str(stop_num) + ".npy")
            # plt.plot(plot_time, svr_preds, label="SVR Predictions")
            # plt.plot(plot_time, test_target_UN_stopnum)
            
            plt.fill_between(plot_date, out_UN_mean_stopnum-out_UN_std_stopnum, out_UN_mean_stopnum+out_UN_std_stopnum, color='r', alpha=0.4, label="Prediction Window")
            plt.plot(plot_date, out_UN_mean_stopnum_plot, c='c', label="Predictions")
            # plt.vlines([datetime.datetime.fromisoformat('2018-05-15T08:00:00'), ymin=inf, ymax])
            # plt.axvline([datetime.datetime.fromisoformat('2018-05-15T08:00:00')])
            plt.subplots_adjust(bottom=0.17)
            plt.xlim([plot_date[0]-100*interval_length, plot_date[-1]+100*interval_length])
            plt.xticks(rotation = 30, ha="right")
            # plt.title(f"Flow for Sensor {stop_num} in Set Bravo Plus")
            plt.xlabel("Date")
            plt.ylabel("Flow (vehicles/hour)")
            plt.title(f"Sensor {stop_num} Flow Prediction Distribution")
            # plt.fill_between(range(ex_test_target_UN.shape[0]), ex_test_target_UN[:, stop_num, time_step], out_UN_mean[:, stop_num, time_step])
            plt.legend()
            # plt.show()
            
            plt.subplots_adjust(bottom=0.17)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M %d/%m/%Y'))
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
            plt.xlim([plot_date[0]-100*interval_length, plot_date[-1]+100*interval_length])
            plt.xticks(rotation = 30, ha="right")
            plt.title(f"Standard Deviation of our Distribution for Node {stop_num}")
            plt.xlabel("Date")
            # plt.ylabel("Standard Deviation (vehicles/hour)")
            # plt.ylabel("Relative Standard Deviation (no unit)")
            plt.plot(plot_date, out_UN_std_stopnum)
            # plt.show()
            
            # num_in = np.where(test_target_UN_stopnum > out_UN_mean_stopnum+out_UN_std_stopnum)
            
            # abs_nodist_error = np.abs(test_target_UN_stopnum - nodist_preds)
            # abs_stgcn_error = np.abs(test_target_UN_stopnum - out_UN_mean[:, stop_num, 0])
            # plt.plot(plot_time, abs_nodist_error)
            # plt.plot(plot_time, abs_stgcn_error)
            # plt.show()
            
            # plt.plot(plot_time, abs_svr_error - abs_stgcn_error)
            # plt.show()
            # for i in len(out_UN)
            
            print(f"\nStop number:\t{stop_num}")
            mse, mae, mape, rmse, ev = get_results(test_target_UN_stopnum, out_UN_mean_stopnum)
            
            mses.append(mse)
            maes.append(mae)
            mapes.append(mape)
            rmses.append(rmse)
            evs.append(ev)
            frac_ins1.append(frac_in1)
            frac_ins2.append(frac_in2)
            frac_ins3.append(frac_in3)
            
            print('Average Number of Vehicles: ', np.mean(np.array(test_target_UN_stopnum)))
            print('STDDEV of Vehicles: ', np.std(np.array(test_target_UN_stopnum)))
            avgs.append(np.mean(np.array(test_target_UN_stopnum)))
            stddevs.append(np.std(np.array(test_target_UN_stopnum)))
            
        mse_avg = np.mean(np.array(mses))
        mae_avg = np.mean(np.array(maes))
        mape_avg = np.mean(np.array(mapes))
        rmse_avg = np.mean(np.array(rmses))
        ev_avg = np.mean(np.array(evs))
        avg_avg = np.mean(np.array(avgs))
        stddev_avg = np.mean(np.array(stddevs))
        frac_in_avg1 = np.mean(np.array(frac_ins1))
        frac_in_avg2 = np.mean(np.array(frac_ins2))
        frac_in_avg3 = np.mean(np.array(frac_ins3))
        
        mse_std = np.std(np.array(mses))
        mae_std = np.std(np.array(maes))
        mape_std = np.std(np.array(mapes))
        rmse_std = np.std(np.array(rmses))
        ev_std = np.std(np.array(evs))
        avg_std = np.std(np.array(avgs))
        stddev_std = np.std(np.array(stddevs))
        frac_in_std1 = np.mean(np.array(frac_ins1))
        frac_in_std2 = np.mean(np.array(frac_ins2))
        frac_in_std3 = np.mean(np.array(frac_ins3))

        
        mses.append(mse_avg)
        maes.append(mae_avg)
        mapes.append(mape_avg)
        rmses.append(rmse_avg)
        evs.append(ev_avg)
        avgs.append(avg_avg)
        stddevs.append(stddev_avg)
        frac_ins1.append(frac_in_avg1)
        frac_ins2.append(frac_in_avg2)
        frac_ins3.append(frac_in_avg3)
        
        mses.append(mse_std)
        maes.append(mae_std)
        mapes.append(mape_std)
        rmses.append(rmse_std)
        evs.append(ev_std)
        avgs.append(avg_std)
        stddevs.append(stddev_std)
        frac_ins1.append(frac_in_std1)
        frac_ins2.append(frac_in_std2)
        frac_ins3.append(frac_in_std3)
        
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
        df["FRAC_IN_1"] = frac_ins1
        df["FRAC_IN_2"] = frac_ins2
        df["FRAC_IN_3"] = frac_ins3
        
        
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

        print("\nResults together")
    
        mse, mae, mape, rmse, ev = get_results(test_target_UN[:, :, 0], out_UN_mean[:, :])
        
        may7 = np.arange(0, 480)
        may14 = np.arange(480*7, 480*8)
        may1213 = np.arange(480*5, 480*7)
        weekdays = np.append(np.arange(480, 480*5), np.arange(480*7, 480*9))
        
        print("\nJust for May 7")
        mse, mae, mape, rmse, ev = get_results(test_target_UN[may7, :, 0], out_UN_mean[may7, :])
        
        print("\nJust for May 14")
        mse, mae, mape, rmse, ev = get_results(test_target_UN[may14, :, 0], out_UN_mean[may14, :])
        
        print("\nJust for May 12 and 13")
        mse, mae, mape, rmse, ev = get_results(test_target_UN[may1213, :, 0], out_UN_mean[may1213, :])
        
        print("\nWeekdays")
        mse, mae, mape, rmse, ev = get_results(test_target_UN[weekdays, :, 0], out_UN_mean[weekdays, :])

        day_mses, day_maes, day_mapes, day_rmses, day_evs, day_avgs, day_stddevs = [], [], [], [], [], [], []

        for day in range(9):
            print(f"Results for Day {day}")
            day_range = np.arange(480*day, 480*(day+1))
            day_mse, day_mae, day_mape, day_rmse, day_ev = get_results(test_target_UN[day_range, :, 0], out_UN_mean[day_range, :])
            
            day_mses.append(day_mse)
            day_maes.append(day_mae)
            day_mapes.append(day_mape)
            day_rmses.append(day_rmse)
            day_evs.append(day_ev)
            
            all_stops_that_day = test_target_UN[day_range, :, 0]
            
            day_avgs.append(np.mean(np.array(test_target_UN[day_range, :, 0])))
            day_stddevs.append(np.std(np.array(test_target_UN[day_range, :, 0])))

        df = pd.DataFrame()
        
        node_list = list(range(9))
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
        
        df.to_csv(folder_string + "/day_results.csv")