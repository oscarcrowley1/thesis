import os
import zipfile
import numpy as np
import torch


def load_scats_data():
    # if (not os.path.isfile("STGCN-PyTorch-master/data/adj_mat.npy")
    #         or not os.path.isfile("STGCN-PyTorch-master/data/node_values.npy")):
    #     with zipfile.ZipFile("STGCN-PyTorch-master/data/METR-LA.zip", 'r') as zip_ref:
    #         zip_ref.extractall("STGCN-PyTorch-master/data/")

    # A = np.load("STGCN-PyTorch-master/data/adj_mat.npy")
    # X = np.load("STGCN-PyTorch-master/data/node_values.npy").transpose((1, 2, 0))
    # X = X.astype(np.float32)
    
    if (not os.path.isfile("STGCN-PyTorch-master/data/adj_mat_alpha.npy")
            or not os.path.isfile("STGCN-PyTorch-master/data/node_values_alpha.npy")):
        with zipfile.ZipFile("STGCN-PyTorch-master/data/SCATS.zip", 'r') as zip_ref:
            zip_ref.extractall("STGCN-PyTorch-master/data/")
    
    A = np.load("STGCN-PyTorch-master/data/interpret_csv/adj_mat_alpha.npy")
    A = A.astype(np.float32)
    X = np.load("STGCN-PyTorch-master/data/interpret_csv/node_values_alpha.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # print(X.shape)
    # X = X[:, 0, :] # Flow only for predictions
    # X = np.expand_dims(X, axis=1)
    # print(X.shape)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    print(f"Setup Info")
    print(f"Adjacency Shape:\t{A.shape}")
    print(f"Data Shape:\t{X.shape}")
    print(f"Num Stops:\t{X.shape[0]}")
    print(f"Num Channels:\t{X.shape[1]}")
    print(f"Num Days:\t{X.shape[2] / 1440}")

    return A, X, means, stds


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
