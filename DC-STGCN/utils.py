from calendar import weekday
import os
import zipfile
import numpy as np
import torch
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def print_save(file, string_to_use):
    print(string_to_use)
    file.write(string_to_use + "\n")

def load_scats_data(dataset):
    # if (not os.path.isfile("DC-STGCN/data/adj_mat.npy")
    #         or not os.path.isfile("DC-STGCN/data/node_values.npy")):
    #     with zipfile.ZipFile("DC-STGCN/data/METR-LA.zip", 'r') as zip_ref:
    #         zip_ref.extractall("DC-STGCN/data/")

    # A = np.load("DC-STGCN/data/adj_mat.npy")
    # X = np.load("DC-STGCN/data/node_values.npy").transpose((1, 2, 0))
    # X = X.astype(np.float32)
    if dataset == "alpha":
        # if (not os.path.isfile("DC-STGCN/data/adj_mat_alpha.npy")
        #         or not os.path.isfile("DC-STGCN/data/node_values_alpha.npy")):
        with zipfile.ZipFile("DC-STGCN/data/SCATS_alpha.zip", 'r') as zip_ref:
            zip_ref.extractall("DC-STGCN/data/")
        
        A = np.load("DC-STGCN/data/alpha_data/adj_mat_alpha.npy")
        A = A.astype(np.float32)
        X = np.load("DC-STGCN/data/alpha_data/node_values_alpha.npy").transpose((1, 2, 0))
        X = X.astype(np.float32)

    elif dataset == "bravo":
        # if (not os.path.isfile("DC-STGCN/data/adj_mat_bravo.npy")
        #         or not os.path.isfile("DC-STGCN/data/node_values_bravo.npy")):
        with zipfile.ZipFile("DC-STGCN/data/SCATS_bravo.zip", 'r') as zip_ref:
            zip_ref.extractall("DC-STGCN/data/")
    
        A = np.load("DC-STGCN/data/bravo_data/adj_mat_bravo.npy")
        A = A.astype(np.float32)
        X = np.load("DC-STGCN/data/bravo_data/node_values_bravo.npy").transpose((1, 2, 0))
        X = X.astype(np.float32)

    elif dataset == "bravoplus":
        # if (not os.path.isfile("DC-STGCN/data/adj_mat_bravo.npy")
        #         or not os.path.isfile("DC-STGCN/data/node_values_bravo.npy")):
        with zipfile.ZipFile("DC-STGCN/data/SCATS_bravoplus.zip", 'r') as zip_ref:
            zip_ref.extractall("DC-STGCN/data/")
    
        A = np.load("DC-STGCN/data/bravoplus_data/adj_mat_bravoplus.npy")
        A = A.astype(np.float32)
        X = np.load("DC-STGCN/data/bravoplus_data/node_values_bravoplus.npy").transpose((1, 2, 0))
        X = X.astype(np.float32)

    else:
        print("EVALUATION DATASET")
        # with zipfile.ZipFile((dataset + "/"), 'r') as zip_ref:
        #     zip_ref.extractall("DC-STGCN/data/")
        junc_set = (dataset.split('/')[-1]).split('_')[0]
    
        A = np.load(dataset + "/" + junc_set + "_data/adj_mat_" + junc_set + ".npy")
        A = A.astype(np.float32)
        X = np.load(dataset + "/" + junc_set + "_data/node_values_" + junc_set + ".npy").transpose((1, 2, 0))
        X = X.astype(np.float32)

    # info_string += X.shape)
    # X = X[:, 0, :] # Flow only for predictions
    # X = np.expand_dims(X, axis=1)
    # info_string += X.shape)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    info_string = "Input Info\n--------------\n"
    info_string += f"Adjacency Shape:\t{A.shape}\n"
    info_string += f"Data Shape:\t{X.shape}\n"
    info_string += f"Num Stops:\t{X.shape[0]}\n"
    info_string += f"Num Channels:\t{X.shape[1]}\n"
    info_string += f"Num Days:\t{X.shape[2] / 480}\n"

    return A, X, means, stds, info_string


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

def define_prev_timesteps(i):
    day_length = 480
    week_length = 480*7
    t_plus = 5
    prev_week_array = [i-week_length, i-week_length+1, i-week_length+2, i-week_length+3, i-week_length+4, i-week_length+5]
    prev_day_array = [i-day_length, i-day_length+1, i-day_length+2, i-day_length+3, i-day_length+4, i-day_length+5]
    prev_immediate_array = [i-12, i-11, i-10, i-9, i-8, i-7, i-6, i-5, i-4, i-3, i-2, i-1, i]
    future_array = [i+5]
    #return [i-week_length, i-day_length, i-12, i-11, i-10, i-9, i-8, i-7, i-6, i-5, i-4, i-3, i-2, i-1, i, i+5]
    return prev_week_array + prev_day_array + prev_immediate_array + future_array

def generate_feature_vects(X):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    # indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
    #            in range(X.shape[2] - (
    #             num_timesteps_input + num_timesteps_output) + 1)]

    distance_back = int(7*24*60/3) # 480*7
    distance_forward = 5
    spread_indices = [define_prev_timesteps(i) for i
               in range(distance_back, X.shape[2] - distance_forward - 1)]


    num_features = len(spread_indices[0]) - 1

    # Save samples
    features, target = [], []

    for index_array in spread_indices:
        features.append(
            X[:, :, index_array[:-1]].transpose(
                (0, 2, 1)))
        #target.append(np.expand_dims(X[:, 0, index_array[-1]], axis=2))
        target_to_append = X[:, 0, index_array[-1]]
        target.append(np.expand_dims(target_to_append, axis=1))
        #target.append(X[:, 0, index_array[-1]])

    total_input = torch.from_numpy(np.array(features))
    total_target = torch.from_numpy(np.array(target))

    test_indx = np.arange((480*36), (480*45)) # day 36 mon to 44 tues inclusive

    before_indx = np.arange((480*36))
    after_indx = np.arange((480*45), total_input.shape[0])

    print(before_indx)
    print(after_indx)

    other_indx = np.concatenate((before_indx, after_indx))
    print(other_indx)
    np.random.shuffle(other_indx)
    split_line = int(other_indx.shape[0] * (5/9))
    training_indx = other_indx[:split_line]
    val_indx = other_indx[split_line:]

    training_input = total_input[training_indx, :, :, :]
    training_target = total_target[training_indx, :, :]

    val_input = total_input[val_indx, :, :, :]
    val_target = total_target[val_indx, :, :]

    test_input = total_input[test_indx, :, :, :]
    test_target = total_target[test_indx, :, :]

    # return torch.from_numpy(np.array(features)), \
    #        torch.from_numpy(np.array(target)), \
    #        num_features

    return training_input, training_target, \
            val_input, val_target, \
            test_input, test_target, \
            num_features

def generate_flow_only_feature_vects(X):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    # indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
    #            in range(X.shape[2] - (
    #             num_timesteps_input + num_timesteps_output) + 1)]

    distance_back = int(7*24*60/3) # 480*7
    distance_forward = 5
    spread_indices = [define_prev_timesteps(i) for i
               in range(distance_back, X.shape[2] - distance_forward - 1)]


    num_features = len(spread_indices[0]) - 1

    # Save samples
    features, target = [], []

    for index_array in spread_indices:
        features.append(
            X[:, :, index_array[:-1]].transpose(
                (0, 2, 1)))
        #target.append(np.expand_dims(X[:, 0, index_array[-1]], axis=2))
        target_to_append = X[:, 0, index_array[-1]]
        target.append(np.expand_dims(target_to_append, axis=1))
        #target.append(X[:, 0, index_array[-1]])

    total_input = torch.from_numpy(np.array(features))
    total_target = torch.from_numpy(np.array(target))

    test_indx = np.arange((480*36), (480*45)) # day 36 mon to 44 tues inclusive

    before_indx = np.arange((480*36))
    after_indx = np.arange((480*45), total_input.shape[0])

    print(before_indx)
    print(after_indx)

    other_indx = np.concatenate((before_indx, after_indx))
    print(other_indx)
    np.random.shuffle(other_indx)
    split_line = int(other_indx.shape[0] * (5/9))
    training_indx = other_indx[:split_line]
    val_indx = other_indx[split_line:]

    training_input = total_input[training_indx, :, :, 0:1]
    training_target = total_target[training_indx, :, :]

    val_input = total_input[val_indx, :, :, 0:1]
    val_target = total_target[val_indx, :, :]

    test_input = total_input[test_indx, :, :, 0:1]
    test_target = total_target[test_indx, :, :]

    # return torch.from_numpy(np.array(features)), \
    #        torch.from_numpy(np.array(target)), \
    #        num_features

    return training_input, training_target, \
            val_input, val_target, \
            test_input, test_target, \
            num_features


def generate_density_only_feature_vects(X):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    # indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
    #            in range(X.shape[2] - (
    #             num_timesteps_input + num_timesteps_output) + 1)]

    distance_back = int(7*24*60/3) # 480*7
    distance_forward = 5
    spread_indices = [define_prev_timesteps(i) for i
               in range(distance_back, X.shape[2] - distance_forward - 1)]


    num_features = len(spread_indices[0]) - 1

    # Save samples
    features, target = [], []

    for index_array in spread_indices:
        features.append(
            X[:, :, index_array[:-1]].transpose(
                (0, 2, 1)))
        #target.append(np.expand_dims(X[:, 0, index_array[-1]], axis=2))
        target_to_append = X[:, 0, index_array[-1]]
        target.append(np.expand_dims(target_to_append, axis=1))
        #target.append(X[:, 0, index_array[-1]])

    total_input = torch.from_numpy(np.array(features))
    total_target = torch.from_numpy(np.array(target))

    test_indx = np.arange((480*36), (480*45)) # day 36 mon to 44 tues inclusive

    before_indx = np.arange((480*36))
    after_indx = np.arange((480*45), total_input.shape[0])

    print(before_indx)
    print(after_indx)

    other_indx = np.concatenate((before_indx, after_indx))
    print(other_indx)
    np.random.shuffle(other_indx)
    split_line = int(other_indx.shape[0] * (5/9))
    training_indx = other_indx[:split_line]
    val_indx = other_indx[split_line:]

    training_input = total_input[training_indx, :, :, 1:2]
    training_target = total_target[training_indx, :, :]

    val_input = total_input[val_indx, :, :, 1:2]
    val_target = total_target[val_indx, :, :]

    test_input = total_input[test_indx, :, :, 1:2]
    test_target = total_target[test_indx, :, :]

    # return torch.from_numpy(np.array(features)), \
    #        torch.from_numpy(np.array(target)), \
    #        num_features

    return training_input, training_target, \
            val_input, val_target, \
            test_input, test_target, \
            num_features

# A, X, means, std, info = load_scats_data("alpha")
# training_input, training_target, val_input, val_target, test_input, test_target, num_timesteps_input = generate_flow_only_feature_vects(X)
# print("TRI:\t{}\nTRT:\t{}\nVI:\t{}\nVT:\t{}\nTEI:\t{}\nTET:\t{}\n".format(training_input.shape, training_target.shape, val_input.shape, val_target.shape, test_input.shape, test_target.shape))


def generate_weekday_feature_vects(X):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    # indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
    #            in range(X.shape[2] - (
    #             num_timesteps_input + num_timesteps_output) + 1)]

    distance_back = int(7*24*60/3) # 480*7
    distance_forward = 5
    spread_indices = [define_prev_timesteps(i) for i
               in range(distance_back, X.shape[2] - distance_forward - 1)]


    num_features = len(spread_indices[0]) - 1

    # Save samples
    features, target = [], []

    for index_array in spread_indices:
        features.append(
            X[:, :, index_array[:-1]].transpose(
                (0, 2, 1)))
        #target.append(np.expand_dims(X[:, 0, index_array[-1]], axis=2))
        target_to_append = X[:, 0, index_array[-1]]
        target.append(np.expand_dims(target_to_append, axis=1))
        #target.append(X[:, 0, index_array[-1]])

    total_input = torch.from_numpy(np.array(features))
    total_target = torch.from_numpy(np.array(target))
    
    

    test_indx = np.concatenate((np.arange((480*36), (480*41)), np.arange((480*43), (480*45)))) # day 36 mon to 44 tues inclusive

    before_indx = np.concatenate((np.arange((480*1), (480*6)), np.arange((480*8), (480*13)), np.arange((480*15), (480*20)), np.arange((480*22), (480*27)), np.arange((480*30), (480*34)))) # bank holidays day 29 and 57
    after_indx = np.concatenate((np.arange((480*45), (480*48)), np.arange((480*50), (480*55)), np.arange((480*58), (480*62)), np.arange((480*64), (480*69)), np.arange((480*71), total_input.shape[0])))
    # after_indx = np.arange((480*45), total_input.shape[0])

    print(before_indx)
    print(after_indx)

    other_indx = np.concatenate((before_indx, after_indx))
    print(other_indx)
    np.random.shuffle(other_indx)
    split_line = int(other_indx.shape[0] * (5/9))
    training_indx = other_indx[:split_line]
    val_indx = other_indx[split_line:]
    
    # plt.scatter(test_indx, test_indx, label="test")
    # plt.scatter(training_indx, training_indx, label="train")
    # plt.scatter(val_indx, val_indx, label="val")
    # plt.legend()
    # plt.show()

    training_input = total_input[training_indx, :, :, :]
    training_target = total_target[training_indx, :, :]

    val_input = total_input[val_indx, :, :, :]
    val_target = total_target[val_indx, :, :]

    test_input = total_input[test_indx, :, :, :]
    test_target = total_target[test_indx, :, :]

    # return torch.from_numpy(np.array(features)), \
    #        torch.from_numpy(np.array(target)), \
    #        num_features

    return training_input, training_target, \
            val_input, val_target, \
            test_input, test_target, \
            num_features

def get_results(y_true, y_pred): # produces metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mean_absolute_percentage_error=metrics.mean_absolute_percentage_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print('MSE: ', round((mse),4))    
    print('MAE: ', round(mean_absolute_error,4))
    print('MAPE: ', round(mean_absolute_percentage_error,4))    
    print('RMSE: ', round(rmse,4))
    print('explained_variance: ', round(explained_variance,4))
    return mse, mean_absolute_error, mean_absolute_percentage_error, rmse, explained_variance


# A, X, means, stds, info = load_scats_data("alpha")
# weekday_vects = generate_weekday_feature_vects(X)