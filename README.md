# thesis
Predicting Dublin Traffic with a Spatio-Temporal Graph Convolution Network

Using SCATS data from Dublin City Council to predict traffic flow.
The STGCN consists of two spatial convolution blocks with a temporal block in between.
It takes two inputs: the adjacency matrix and the node values.
The adjacency matrix is (# sensors)*(# sensors)
The node value tensor is (# timesteps)*(# sensors)*(# channels).
We will be using two channels; flow and density
