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
from dataset.qm9 import QM9Properties
from dataset.utils import get_mean_and_mad
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.utils import smiles
from rdkit.Chem import rdchem
from torch_geometric.data import Data


def split_data(data):
    
    
    train_split = data[:100000]
    dev_split = data[100000:118000]
    test_split = data[118000:]

    return train_split, dev_split, test_split


'''
It returns for all the features 0 so we will not use it for now. There is something called smiles
that can be used somehow to get those.

def get_atom_features(atom):
    # atomic_num = atom.GetAtomicNum()
    hybridization = atom.GetHybridization().real
    is_aromatic = atom.GetIsAromatic()
    chiral_tag = atom.GetChiralTag().real
    formal_charge = atom.GetFormalCharge()

    return [hybridization, is_aromatic, chiral_tag, formal_charge]

def mol_from_data(data):
    mol = rdchem.EditableMol(rdchem.Mol())

    # Add atoms
    for z in data.z:
        mol.AddAtom(rdchem.Atom(int(z)))

    # Add bonds
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        atom1, atom2 = edge_index[:, i]
        if atom1 < atom2:
            bond_type = rdchem.BondType.SINGLE
            mol.AddBond(int(atom1), int(atom2), bond_type)

    # Finalize the molecule
    mol = mol.GetMol()
    return mol



def compute_extended_features(data):
    mol = Chem.RWMol()  # Create an empty editable molecule

    # Add atoms to the molecule
    for charge in data.z.tolist():
        atom = Chem.Atom(int(charge))
        mol.AddAtom(atom)

    # Add bonds to the molecule
    for edge_indices in data.edge_index.transpose(0, 1).tolist():
        atom1, atom2 = edge_indices
        if not mol.GetBondBetweenAtoms(atom1, atom2):
            mol.AddBond(atom1, atom2, Chem.rdchem.BondType.SINGLE)

    # Compute the features for each atom in the molecule
    atom_features = []

    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    return atom_features

'''




datasets = {
    "QM9": QM9
}

dataset_properties = {
    "QM9": QM9Properties
}

pools = {
    "add": global_add_pool,
    "mean": global_mean_pool
}

act_fns = {
    "SiLU": nn.SiLU
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EGNN Training Script')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
    parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='number of workers for the dataloader')
    parser.add_argument("--dataset", type=str, default="QM9", 
                        help="Dataset to use (QM9, )")
    parser.add_argument("--property", type=str, default="ALPHA", 
                        help="Property to predict (QM9: MU, ALPHA, ...)")
    parser.add_argument("--pooling", type=str, default="add", 
                        help="Pooling method (add, mean)")
    parser.add_argument("--act_fn", type=str, default="SiLU", 
                        help="Activation function (SiLU)")
    parser.add_argument("--aggregation", type=str, default="add", 
                        help="Aggregation method for message passing (default(add), mean, max)")           
    parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                        help='learning rate')
    parser.add_argument('--radius', type=float, default=None, metavar='N',
                        help='Used when working with Radius Graph (Default None)')
    parser.add_argument('--dim', type=int, default=3, metavar='N',
                        help='Coordinate Dimension')
    parser.add_argument('--node_feature_s', type=int, default=11, metavar='N',
                        help='Node Feature size')
    parser.add_argument('--hidden_feature_s', type=int, default=128, metavar='N',
                        help='Hidden Feature size')
    parser.add_argument('--out_feature_s', type=int, default=1, metavar='N',
                        help='Output Feature size')
    parser.add_argument('--num_layers', type=int, default=7, metavar='N',
                        help='Number of Layers')
    parser.add_argument("--wandb_project_name", type=str, default="LSPE-EGNN", 
                        help="Project name for Wandb")
    parser.add_argument("--accelerator", type=str, default="cpu", 
                        help="Type of Hardware to run on (cpu, gpu, tpu, ...)")
    

    args = parser.parse_args()

    assert args.pooling in pools
    assert args.act_fn in act_fns

    assert args.dataset in datasets
    assert args.dataset in dataset_properties
    
    properties = dataset_properties[args.dataset]
    dataset_class = datasets[args.dataset]

    assert args.property in properties._member_names_
    args.property = properties[args.property]

    assert args.train_split + args.val_split + args.test_split == 1.0
    
    
    pl.seed_everything(args.seed, workers=True)

    print("Obtaining Dataset")
    
    dataset = dataset_class(root = args.dataset_path).shuffle()


    # for data in dataset:   all zeros--> deleted for now. used to get the extra features
    #     extended_features = torch.tensor(compute_extended_features(data))
    #     data.x = torch.cat([data.x, extended_features], dim=1)

    print("Creating Data Splits")

    train_data, valid_data, test_data = split_data(dataset)
    print("Creating DataLoaders")

    train_loader = DataLoader(train_data, batch_size = args.batch_size, num_workers = args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, num_workers = args.num_workers)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, num_workers = args.num_workers)
    
    print("Computing Mean & Mad")
    mean, mad = get_mean_and_mad(train_loader, args.property)

    print("Creating Model")

    model = EGNN(args.node_feature_s, args.hidden_feature_s, args.out_feature_s, 
                args.num_layers, args.dim, args.radius, aggr = args.aggregation, act=act_fns[args.act_fn], pool=pools[args.pooling])
    
    if isinstance(dataset, QM9):
        model = QM9Regressor(model, args.property, lr=args.lr, weight_decay=args.weight_decay, mean=mean, mad=mad, epochs = args.epochs)
    else:
        #If we implement multiple datasets we can add more type checks
        raise NotImplementedError


    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project=args.wandb_project_name)
    # Length of data
    wandb_logger.experiment.config["Length of Train Data"] = len(train_data)
    wandb_logger.experiment.config["Length of Dev Data"] = len(valid_data)
    wandb_logger.experiment.config["Length of Test Data"] = len(test_data)

    # add your batch size to the wandb config
    wandb_logger.experiment.config["dataset"] = args.dataset
    wandb_logger.experiment.config["property"] = args.property.name

    
    trainer = pl.Trainer(logger=wandb_logger, accelerator=args.accelerator, max_epochs=args.epochs)
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

    wandb.finish()