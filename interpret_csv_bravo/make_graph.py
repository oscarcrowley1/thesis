from cmath import inf
from turtle import color
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

adj_matrix = np.load("interpret_csv/adj_mat_alpha_0603.npy")

#adj_matrix = np.transpose(adj_matrix)

graph = nx.from_numpy_matrix(adj_matrix, parallel_edges=True, create_using=nx.DiGraph)

#graph = nx.reverse(graph)

# for from_node in range(adj_matrix)

info = [weight for fromx, tox, weight in graph.edges.data("weight")]
print(info)
print(len(info))


# node_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
node_labels = {
    "Node1": 0,
    "Node2": 1,
    "Node3": 2,
    "Node4": 3,
    "Node5": 4,
    "Node6": 5,
    "Node7": 6,
    "Node8": 7,
    "Node9": 8,
}

node_pos = {
    0: (1,0.25),
    1: (1,1.5),
    2: (-3,-1),
    3: (-3,1.5),
    4: (-2,0.25),
    5: (3,-4),
    6: (4,-3),
    7: (4,3.5),
    8: (3,4.5),
}

plt.set_cmap(plt.get_cmap("YlGnBu"))

graph_drawing = nx.draw_networkx(graph, pos=node_pos, with_labels=True, arrows=True, edge_color=info, edge_cmap=plt.get_cmap("YlGnBu"))
dummy_plot = plt.scatter(inf, inf, cmap="Blues") # plt.get_cmap("YlGnBu"))

# heatmap = plt.pcolor(info,cmap=plt.get_cmap("viridis"))

plt.title("Kernelised Graph Structure")
plt.colorbar(dummy_plot)# plt.get_cmap("YlGnBu"))
plt.show()
