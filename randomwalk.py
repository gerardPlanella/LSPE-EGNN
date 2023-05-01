import numpy as np

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
    
    #conduct random walk by using the property that the nth-potency of the adjacency matrix is a "number of connections in n steps" matrix
    randomwalk_matrix = []
    for i in range(1,N+1):
        randomwalk_i = np.linalg.matrix_power(adj_matrix.copy(), i)
        print(randomwalk_i)
        randomwalk_i /= np.sum(randomwalk_i, axis=0)
        randomwalk_matrix.append(randomwalk_i)

    #the graph laplacian: the degree matrix - adjacancy matrix
    laplacian = np.diag(np.sum(adj_matrix, axis=0)) - adj_matrix


    return randomwalk_matrix, laplacian

