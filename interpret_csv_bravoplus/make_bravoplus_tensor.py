from calendar import weekday
import os
import zipfile
import numpy as np
import torch
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# if dataset == "alpha":
        # if (not os.path.isfile("STGCN-PyTorch-master/data/adj_mat_alpha.npy")
        #         or not os.path.isfile("STGCN-PyTorch-master/data/node_values_alpha.npy")):
with zipfile.ZipFile("STGCN-PyTorch-master/data/SCATS_alpha.zip", 'r') as zip_ref:
    zip_ref.extractall("STGCN-PyTorch-master/data/")

X_alpha = np.load("STGCN-PyTorch-master/data/alpha_data/node_values_alpha.npy").transpose((1, 2, 0))
X_alpha = X_alpha.astype(np.float32)

    # elif dataset == "bravo":
        # if (not os.path.isfile("STGCN-PyTorch-master/data/adj_mat_bravo.npy")
        #         or not os.path.isfile("STGCN-PyTorch-master/data/node_values_bravo.npy")):
with zipfile.ZipFile("STGCN-PyTorch-master/data/SCATS_bravo.zip", 'r') as zip_ref:
    zip_ref.extractall("STGCN-PyTorch-master/data/")

# A = np.load("STGCN-PyTorch-master/data/bravo_data/adj_mat_bravo.npy")
# A = A.astype(np.float32)
X_bravo = np.load("STGCN-PyTorch-master/data/bravo_data/node_values_bravo.npy").transpose((1, 2, 0))
X_bravo = X_bravo.astype(np.float32)

print(X_alpha.shape)
print(X_bravo.shape)

X_196 = X_alpha[2:5, :, :]
print(X_196.shape)

X_bravoplus = np.concatenate((X_bravo, X_196), axis=0)
print(X_bravoplus.shape)

X_bravoplus = X_bravoplus.transpose((2, 0, 1))

np.save("interpret_csv_bravoplus/node_values_bravoplus", X_bravoplus)

files_string = "TO BE CONFIGURED"

f = open("interpret_csv_bravoplus/nv_info.txt", "w")
info_string = "Num Juncs:\t" + str(X_bravoplus.shape[1]) + "\nNum Channels:\t" + str(X_bravoplus.shape[2]) + "\nNum Days:\t" + str(X_bravoplus.shape[0]/480)
print(info_string)
f.write(info_string)
f.write(files_string)
f.close()

if os.path.isfile("interpret_csv_bravoplus/adj_mat_bravoplus.npy") and os.path.isfile("interpret_csv_bravoplus/adj_info.txt"):
    with zipfile.ZipFile("interpret_csv_bravoplus/SCATS_bravoplus.zip", "w") as zip_object:
        zip_object.write("interpret_csv_bravoplus/node_values_bravoplus.npy", arcname="bravoplus_data/node_values_bravoplus.npy")
        zip_object.write("interpret_csv_bravoplus/adj_mat_bravoplus.npy", arcname="bravoplus_data/adj_mat_bravoplus.npy")
        zip_object.write("interpret_csv_bravoplus/adj_info.txt", arcname="bravoplus_data/adj_info.npy")
        zip_object.write("interpret_csv_bravoplus/nv_info.txt", arcname="bravoplus_data/nv_info.npy")
    print("Zipped")