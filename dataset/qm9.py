import torch
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.datasets import QM9
from torch_geometric.transforms import RadiusGraph, Compose, BaseTransform, Distance, Cartesian, RandomRotate
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils





class QM9Transform(BaseTransform):
    def _init_(self):
        self.cartesian = Cartesian()
        self.distance = Distance(norm=False)

    def _call_(self, data):
        # Convert the molecular graph to Cartesian coordinates
        data = self.cartesian(data)

        # Calculate the pairwise distances between atoms
        data = self.distance(data)

        # Convert the distance matrix to edge indices and node features
        edge_index, dist = utils.dense_to_sparse(data.dist)
        data.edge_index = edge_index.t().contiguous()

        # Add the atomic number as node features
        data.x = torch.tensor(data.atomic_numbers, dtype=torch.long)

        return data

    def collate(self, data_list):
        return utils.batch(data_list)


if __name__ == "__main__":
    dataset = QM9(root = "./data")
    loader = DataLoader(dataset, batch_size = 1)
    for batch in loader:
        for item in batch:
            print("hello")
            break
        break

