import numpy as np
import torch
import scipy.sparse as sp
from model import GMCM_VGAE
from preprocessing import load_data, sparse_to_tuple, preprocess_graph
import time
from random import randint
import math

# Code below is adapted from https://github.com/nairouz/R-GAE/tree/master/GMM-VGAE. We thank for the authors to make it publicly available 

dataset = "ExampleData"
nClusters = 14
adj, features, labels = load_data('ExampleData', './data/ExampleData', True)

features_new = features.toarray()
num_nodes = features.shape[0]
num_features = features.shape[1]

# Network hyperparameter
embedding_size = 45
num_neurons = 90
save_path = "./results/"

# Clustering hyperparameters
epochs_cluster = 200
lr_cluster = 0.01

# Configure the device to cuda
torch.set_default_tensor_type('torch.cuda.FloatTensor') 
device = torch.device("cuda")
print(torch.cuda.is_available())

# Data processing 
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.cuda.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T).cuda(), torch.FloatTensor(adj_norm[1]).cuda(), torch.Size(adj_norm[2]))
adj_label = torch.cuda.sparse.FloatTensor(torch.LongTensor(adj_label[0].T).cuda(), torch.FloatTensor(adj_label[1]).cuda(), torch.Size(adj_label[2]))
features = torch.cuda.sparse.FloatTensor(torch.LongTensor(features[0].T).cuda(), torch.FloatTensor(features[1]).cuda(), torch.Size(features[2]))

weight_mask_orig = adj_label.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig

print("start")
# Start the timer
start = time.perf_counter()
# Training
acc_array = []

ress = []
seed = 42 

network = GMCM_VGAE(adj = adj_norm , num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters=nClusters, activation="Sigmoid", seed=seed)
network.to(device)
res, y_pred, y = network.train([], adj_norm, features, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_cluster, lr=lr_cluster, save_path=save_path, dataset=dataset, features_new=features_new)
end = time.perf_counter()

print(f"Total time: {end - start:0.4f} seconds")
    