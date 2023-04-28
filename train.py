import argparse
import sys

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.datasets import QM9
from torch_geometric.transforms import RadiusGraph, Compose, BaseTransform, Distance, Cartesian, RandomRotate
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
import wandb

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
    parser = argparse.ArgumentParser(description='EGNN Training Script')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--dataset_path', type=str, default='./data', metavar='N',
                        help='Path to save dataset')
    parser.add_argument('--train_split', type=float, default=0.6, metavar='S',
                        help='Percentage of Data to use for training (default: 0.6)')
    parser.add_argument('--val_split', type=float, default=0.2, metavar='S',
                        help='Percentage of Data to use for validation (default: 0.6)')
    parser.add_argument('--test_split', type=float, default=0.2, metavar='S',
                        help='Percentage of Data to use for testing (default: 0.6)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers for the dataloader')
    #TODO: Implement change in dataset
    parser.add_argument("--dataset", type=str, default="QM9", 
                        help="Dataset to use (QM9, )")
    parser.add_argument("--property", type=str, default="ALPHA", 
                        help="Property to predict (QM9: MU, ALPHA, ...)")
    parser.add_argument("--aggregation", type=str, default="add", 
                        help="Aggregation method for message passing (default(add), mean, max)")           
    parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='learning rate')
    parser.add_argument('--node_feature_s', type=int, default=11, metavar='N',
                        help='Node Feature size')
    parser.add_argument('--hidden_feature_s', type=int, default=128, metavar='N',
                        help='Hidden Feature size')
    parser.add_argument('--out_feature_s', type=int, default=1, metavar='N',
                        help='Output Feature size')
    parser.add_argument('--num_layers', type=int, default=4, metavar='N',
                        help='Number of Layers')
    parser.add_argument("--wandb_project_name", type=str, default="LSPE-EGNN", 
                        help="Project name for Wandb")
    parser.add_argument("--accelerator", type=str, default="cpu", 
                        help="Type of Hardware to run on (cpu, gpu, tpu, ...)")
    

    

    args = parser.parse_args()

    assert args.train_split + args.val_split + args.test_split <= 1.0
    
    if args.dataset == "QM9":
        assert args.property in QM9Properties._member_names_
        args.property = QM9Properties[args.property]


    pl.seed_everything(args.seed, workers=True)

    dataset = QM9(root = args.dataset_path).shuffle()

    train_data, valid_data, test_data = split_data(dataset, args.train_split, args.val_split, args.test_split)

    train_loader = DataLoader(train_data, batch_size = args.batch_size, num_workers = args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, num_workers = args.num_workers)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, num_workers = args.num_workers)
    
    mean, mad = get_mean_and_mad(train_loader, args.property)

    #TODO: Do we want to add arguments for the activation fn, dim and pool?
    model = EGNN(args.node_feature_s, args.hidden_feature_s, args.out_feature_s, 
                args.num_layers, 3, None, aggr = args.aggregation, act=nn.SiLU, pool=global_add_pool)
    model = QM9Regressor(model, args.property, lr=args.lr, weight_decay=args.weight_decay, mean=mean, mad=mad, epochs = args.epochs)


    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project=args.wandb_project_name)

    # add your batch size to the wandb config
    wandb_logger.experiment.config["dataset"] = args.dataset
    wandb_logger.experiment.config["property"] = args.property.name

    
    trainer = pl.Trainer(logger=wandb_logger, accelerator=args.accelerator, max_epochs=args.epochs)
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

    wandb.finish()