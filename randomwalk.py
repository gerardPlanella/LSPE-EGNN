import numpy as np
import torch

def randomwalk(graph, N = 10):
    #Input: one graph, max depth of random walk
    #Output: 3D random walk matrix of depth N with the random walks stacked ontop of each other, Laplacian
    
    #turn edge index into adjacency matrix 
    edge_index = graph["edge_index"].numpy() 
    adj_matrix = np.zeros((np.max(edge_index)+1, np.max(edge_index)+1))
    for i in range(edge_index.shape[1]):
        start_node, end_node = edge_index[:, i]
        adj_matrix[start_node, end_node] = 1
        adj_matrix[end_node, start_node] = 1
    
    #conduct random walk 
    #degree matrix as row vector:
    degrees = np.sum(adj_matrix, axis=0)
    randomwalk_matrix = [] 
    #walk n=1 is just adjacency matrix / degrees vector
    randomwalk_matrix.append(adj_matrix/degrees)
    for i in range(N):
        randomwalk_i = np.zeros(adj_matrix.shape)
        #divide probabilities by number of outgoing edges
        values = (randomwalk_matrix[-1].T/degrees).T
        #redistribute position probabilities by the edges
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                col_vector = adj_matrix[:, j] * values[i, j]
                randomwalk_i[:, i] += col_vector
        randomwalk_matrix.append(randomwalk_i)

    #the graph laplacian: the degree matrix - adjacancy matrix
    laplacian = np.diag(np.sum(adj_matrix, axis=0)) - adj_matrix


    return torch.from_numpy(randomwalk_matrix), torch.from_numpy(laplacian)

"""
#example graph from https://en.wikipedia.org/wiki/Laplacian_matrix
example=np.array([[0,0,1,1,2,3,3],[1,4,4,2,3,4,5]])
randomwalk(example)
"""
