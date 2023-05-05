import numpy as np
import torch

def randomwalk(graph, N=10):
    """
    Conducts a random walk on a given graph and returns the random walk matrix and Laplacian matrix.
    
    Args:
        graph (dict): A dictionary containing the graph information with 'edge_index' as the key.
                      The 'edge_index' should be a 2xM array where M is the number of edges.
        N (int, optional): The maximum depth of the random walk. Defaults to 10.
        
    Returns:
        tuple: A tuple containing the following:
            - torch.Tensor: A 2D tensor of shape (num_nodes, N) containing the random walk matrix for ending up at the same node at different depths N.
            - torch.Tensor: A 2D tensor of shape (num_nodes, num_nodes) containing the Laplacian matrix of the graph.
    """
    # Convert edge index into adjacency matrix
    edge_index = graph["edge_index"].cpu().numpy()
    adj_matrix = np.zeros((np.max(edge_index)+1, np.max(edge_index)+1))
    for i in range(edge_index.shape[1]):
        start_node, end_node = edge_index[:, i]
        adj_matrix[start_node, end_node] = 1
        adj_matrix[end_node, start_node] = 1

    # Calculate degree matrix as row vector
    degrees = np.sum(adj_matrix, axis=0)
    randomwalk_matrix = []

    # First random walk (n=1) is just adjacency matrix / degrees vector
    randomwalk_matrix.append(adj_matrix/degrees)

    # Conduct random walk for N steps
    for _ in range(N-1):
        randomwalk_i = np.zeros(adj_matrix.shape)

        # Divide probabilities by number of outgoing edges
        values = (randomwalk_matrix[-1].T/degrees).T

        # Redistribute position probabilities by the edges
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                col_vector = adj_matrix[:, j] * values[i, j]
                randomwalk_i[:, i] += col_vector

        randomwalk_matrix.append(randomwalk_i)

    #We want to only know the probabilities of the walk ending up back in the original position for depths k (the diagonals of each random walk matrix)
    #output = np.array([np.diag(matrix) for matrix in randomwalk_matrix]).T

    # Calculate graph Laplacian: degree matrix - adjacency matrix
    laplacian = np.diag(np.sum(adj_matrix, axis=0)) - adj_matrix

    diagonal_elements = np.array([np.diag(rw_matrix) for rw_matrix in randomwalk_matrix]).T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.from_numpy(diagonal_elements).to(device).type(torch.float32), torch.from_numpy(laplacian).to(device).type(torch.float32)



"""
#example graph from https://en.wikipedia.org/wiki/Laplacian_matrix
example=np.array([[0,0,1,1,2,3,3],[1,4,4,2,3,4,5]])
randomwalk(example)
"""
