
import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.datasets import QM9
from torch_geometric.transforms import RadiusGraph, Compose, BaseTransform, Distance, Cartesian, RandomRotate
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
import pytorch_lightning as pl
import sys

from models.egnn import EGNN
from models.regressors import QM9Regressor
from dataset.qm9 import QM9Properties, get_mean_and_mad

def split_data(data, train_percent, dev_percent, test_percent):
    assert train_percent + dev_percent + test_percent == 1.0

    data_len = len(data)
    train_split = data[:int(data_len*train_percent)]
    dev_split = data[int(data_len*train_percent):int(data_len*(train_percent + dev_percent))]
    test_split = data[int(data_len*(train_percent + dev_percent)):]

    return train_split, dev_split, test_split



if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    dataset = QM9(root = "./data").shuffle()
    train_data, valid_data, test_data = split_data(dataset, 0.6, 0.2, 0.2)

    train_loader = DataLoader(train_data, batch_size = 32)
    valid_loader = DataLoader(valid_data, batch_size = 32)
    test_loader = DataLoader(test_data, batch_size = 32)
    
    mean, mad = get_mean_and_mad(train_loader, QM9Properties.ALPHA)

    model = EGNN(11, 128, 1, 4, 3, aggr = "add", act=nn.SiLU, pool=global_add_pool)
    model = QM9Regressor(model, QM9Properties.ALPHA, lr=1e-3, weight_decay=1e-16, mean=mean, mad=mad)

    
    trainer = pl.Trainer(logger=None, accelerator="gpu", max_epochs=1)
    trainer.fit(model, train_loader, valid_loader)