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


def train(model, train_loader, valid_loader, test_loader):





if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = QM9(root = "./data").shuffle()
    train_data, valid_data, test_data = split_data(dataset, 80, 20)

    train_loader = DataLoader(train_data, batch_size = 32)
    valid_loader = DataLoader(train_data, batch_size = 32)
    test_loader = DataLoader(train_data, batch_size = 32)
    
    model = EGNN().to(device)
    
    train(model, train_loader, valid_loader, test_loader)
    
        

