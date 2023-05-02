import numpy as np
import torch
from dataset.qm9 import QM9Properties

def get_mean_and_mad(train_dataset, property):
    
    if isinstance(property, QM9Properties):
        values = []
        n = 0
        for batch in train_dataset:
            values.extend(batch.y[:, property.value])
            
        mean = torch.mean(torch.tensor(values))
        mad = torch.mean(torch.abs(torch.tensor(values) - mean))
        return mean, mad
    else:
        raise NotImplementedError

def lap_positional_encoding(graph, pos_encoding_dim):
    """
    Calculate the graph initial positional encoding using Laplacian Eigenvectors.
    """

    # Calculate the laplacian matrix
    edge_index = graph["edge_index"].numpy()
    num_nodes = np.max(edge_index) + 1
    adj = np.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    deg = np.sum(adj, axis=0)

    # Compute laplacian via factorization of graph laplacian
    lap = np.eye(adj.shape[0]) - (deg ** -0.5) * adj * (deg ** -0.5)

    # Get Eigenvectors
    EigVal, EigVec = np.linalg.eig(lap)
    idx = EigVal.argsort()  # sort increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # Return the eigen vectors up to the set encoding dimension
    return torch.from_numpy(EigVec[:, 1:pos_encoding_dim+1],).float()


# if __name__ == '__main__':
#     example_graph = torch.tensor(data=[[0, 0, 1, 1, 2, 3, 3], [1, 4, 4, 2, 3, 4, 5]])
#     graph = {'edge_index': example_graph}
#     lap_enc = lap_positional_encoding(graph, pos_encoding_dim=10)
#     print(lap_enc)
