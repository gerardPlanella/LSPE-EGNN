
import torch
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.datasets import QM9
from torch_geometric.transforms import RadiusGraph, Compose, BaseTransform, Distance, Cartesian, RandomRotate
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
import sys

from egnn import EGNN

def split_data(data, train_percent, dev_percent, test_percent):
    assert train_percent + dev_percent + test_percent == 1.0

    data_len = len(data)
    train_split = data[:int(data_len*train_percent)]
    dev_split = data[int(data_len*train_percent):int(data_len*(train_percent + dev_percent))]
    test_split = data[int(data_len*(train_percent + dev_percent)):]

    return train_split, dev_split, test_split


def train(train_loader, valid_loader, test_loader):


    for batch in train_loader:
        print("hello")







if __name__ == "__main__":

    sys.path.insert(0, "../")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = QM9(root = "./data").shuffle()
    train_data, valid_data, test_data = split_data(dataset, 0.6, 0.2, 0.2)

    train_loader = DataLoader(train_data, batch_size = 32)
    valid_loader = DataLoader(valid_data, batch_size = 32)
    test_loader = DataLoader(test_data, batch_size = 32)
    
    model = EGNN().to(device)
    

    train(model, train_loader, valid_loader, test_loader)