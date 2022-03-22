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
    df = convert_tb_data("read_tensor_events/Mar03_16-02-17_dropout03")
    # df = convert_tb_data("read_tensor_events/Mar03_15-35-00_dropout05")    # 
    print(df.head())
    
    training_losses = df[df['name'] == "Training Loss"]
    validation_losses = df[df['name'] == "Validation Loss"]
    validation_mae = df[df['name'] == "Validation MAE"]
    validation_losses_unnorm = df[df['name'] == "Validation MSE Unnormalised"]
    
    print(training_losses.head())
    
    tl_array = training_losses['value']
    vl_array = validation_losses['value']
    
    
    plt.plot(np.array(range(len(tl_array))), tl_array, label="Training Loss")
    plt.plot(np.array(range(len(tl_array))), vl_array, label="Validation Loss")
                

    plt.xlabel("Iterations")
    plt.ylabel("Loss on Normalised Values")
    plt.legend()
    plt.show()