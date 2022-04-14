from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.
    
    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.
    
    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.
    
    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.
    
    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )
    
    columns_order = ['wall_time', 'name', 'step', 'value']
    
    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    #print(out[0].head())
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)
        
    return all_df.reset_index(drop=True)

if __name__ == "__main__":
    #dir_path = "/home/kretyn/projects/ai-traineree/runs/"
    #exp_name = "CartPole-v1_2021-01-26_11:02"
    df = convert_tb_data("final_models/bravo_d03_DIST_2503_e15d10_PRUNING_longrun/Apr05_16-53-37_msc-linux-sls-035")
    df2 = convert_tb_data("final_models/bravo_d03_nodist_2503_e15d10_PRUNING/Mar30_13-14-22_msc-linux-sls-035")
    df3 = convert_tb_data("final_models/bravo_d03_i4000_nodist_2503_e15d10_PRUNING/Mar31_20-59-07_msc-linux-sls-035")
    # df = convert_tb_data("read_tensor_events/Mar03_15-35-00_dropout05")    # 
    print(df.head())
    
    training_log_losses = df[df['name'] == "Training Log Loss"]
    validation_log_losses = df[df['name'] == "Validation Log Loss"]
    training_classic_losses = df[df['name'] == "Training Classic Loss"]
    validation_classic_losses = df[df['name'] == "Validation Classic Loss"]
    
    # print(training_losses.head())
    
    tll_array = (np.array(training_log_losses['value'])*300)[0:4000]
    vll_array = (np.array(validation_log_losses['value']))[0:4000]
    
    tcl_array = (np.array(training_classic_losses['value']))[0:4000]
    vcl_array = (np.array(validation_classic_losses['value']))[0:4000]
    
    
    training_classic_losses2 = df2[df2['name'] == "Training Classic Loss"]
    validation_classic_losses2 = df2[df2['name'] == "Validation Classic Loss"]
    
    
    tcl_array2 = np.array(training_classic_losses2['value'])
    vcl_array2 = np.array(validation_classic_losses2['value'])
    
    
    training_classic_losses3 = df3[df3['name'] == "Training Classic Loss"]
    validation_classic_losses3 = df3[df3['name'] == "Validation Classic Loss"]
    
    
    tcl_array3 = np.array(training_classic_losses3['value'])
    vcl_array3 = np.array(validation_classic_losses3['value'])
    
    
    plt.plot(np.array(range(len(tll_array))), tll_array, label="Training")
    plt.plot(np.array(range(len(vll_array))), vll_array, label="Validation")
                

    plt.title("NLP Loss during Training")
    plt.xlabel("Iterations")
    plt.ylabel("NLP Loss")
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    plt.plot(np.array(range(len(vcl_array))), vcl_array, label="Dist Validation", color='b', alpha=0.5)
    
    plt.plot(np.array(range(len(vcl_array2))), vcl_array2, label="No Dist Validation", color='r', alpha=0.5)
    
    plt.plot(np.array(range(len(vcl_array2), len(vcl_array2)+len(vcl_array3))), vcl_array3, color='r', alpha=0.5)
    
    plt.plot(np.array(range(len(tcl_array))), tcl_array, label="Dist Training", color='cyan')
    plt.plot(np.array(range(len(tcl_array2))), tcl_array2, label="No Dist Training", color='yellow')
    plt.plot(np.array(range(len(tcl_array2), len(tcl_array2)+len(tcl_array3))), tcl_array3, color='yellow')

    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("MSE Loss during Training")
    plt.yscale('log')
    plt.legend()
    plt.show()