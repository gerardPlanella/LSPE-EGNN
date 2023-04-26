import torch
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.datasets import QM9
from torch_geometric.transforms import RadiusGraph, Compose, BaseTransform, Distance, Cartesian, RandomRotate
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils

from egnn import EGNN

def split_data(data, train_percent, test_percent):
    assert train_percent + test_percent < 100
    dev_percent = 100 - train_percent - test_percent
    data_len = len(data)
    train_split = data[:data_len*(train_percent/100)]
    dev_split = data[data_len*(train_percent/100):data_len*((train_percent + dev_percent)/100)]
    test_split = data[data_len*((train_percent + dev_percent)/100):]

    return train_split, dev_split, test_split


def train(model, data_split):


if __name__ == "__main__":
    dataset = QM9(root = "./data").shuffle()
    loader = DataLoader(dataset, batch_size = 1)
    for batch in loader:
        for item in batch:

        

