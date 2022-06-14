## Traffic Forecasting in Dublin City using Graph Neural Networks:
# Investigating the use of an STGCN Model in Urban Traffic Flow Prediction

## Abstract
With the implementation of more sophisticated 'smart-city' technologies, traffic data has become more abundant and also more actionable. Despite this, traffic is still a huge problem in most cities, wasting both time and resources, while also damaging our environment. Understanding the spatio-temporal patterns of traffic can help us to forecast more accurately and adjust our infrastructure accordingly. Deep learning has seen huge success in this field of late, due to its ability to find patterns across space and time in non-obvious ways. More recently, Graph Neural Networks (GNNs) have been shown to improve on the state of the art, as they are able to model the innate graph structure of traffic networks. In this project, I apply a Spatio-Temporal Graph Convolutional Network (STGCN) model on Dublin City data in order to predict traffic flow, using flow and density as our inputs. To the best of my knowledge, this is the first time that a Graph Neural Network has been used for urban traffic forecasting in Dublin. I train STGCN models on 3 subsets of traffic junctions in the city centre and compare the performance against a Support Vector Regressor (SVR) model. These STGCN models outperform our baseline in most cases. I also create and evaluate a version of the model that outputs an uncertainty distribution around our estimate. This model was able to accurately represent the uncertainty of its estimates, even on unseen data. Lastly, I create a model that uses flow alone as its input. This one channel model does not perform as well as our two channel model, demonstrating that density plays a role in predicting flow in our experiments.

## Codebase Structure
# OCrowleySchEng2022
Thesis paper detailing design and experiments conducted using this model.
# DC_STGCN
Main folder in which code neural network is created and trained. Includes structure of Spatio-Temporal Graph Convolutional Network used, defined using PyTorch Geometric.
# interpret_csv_X
Converts SCATS text files into numpy tensors which are used by the network.
# analysis
Evaluates model performance on test datasets.
# checkpoints
Stores prgresss during training.
# diagrams
Various diagrams and plots used in thesis.
# final_models
Folder containing the saved models used in our experiments.
# read_tensor_events
Used for interpreting TensorBoard files generated during training.
# runs
Stores TensorBoard files on each training run.
# saved_models
Stores model parameters on each training run.
# svr
Contains SVR model used on baseline. This includes training and testing funcitonality.
# visualise_stgcn
Used to visaulise original data.