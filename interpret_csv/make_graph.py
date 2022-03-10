from turtle import color
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

adj_matrix = np.load("interpret_csv/adj_mat_alpha.npy")

graph = nx.from_numpy_matrix(adj_matrix)

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
    0: (0,0),
    1: (1,1),
    2: (-3,-1),
    3: (-3,1),
    4: (-2,0),
    5: (3.5,-4),
    6: (3.5,-3),
    7: (4,4),
    8: (3,3.5),
}



#graph_drawing = nx.draw_networkx(graph, pos=node_pos, with_labels=True, arrows=True, edge_color=info, edge_cmap=plt.get_cmap("viridis"))
graph_drawing = nx.draw_networkx(graph, arrows=True, arrowstyle='-|>')

# heatmap = plt.pcolor(info,cmap=plt.get_cmap("viridis"))


# plt.colorbar(plt.get_cmap("viridis"))
plt.show()
