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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb

from models.egnn import EGNN
from models.regressors import QM9Regressor
from dataset.qm9 import QM9Properties
from dataset.utils import get_mean_and_mad
from torch_geometric.transforms import RadiusGraph, AddRandomWalkPE, Compose

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from dataset.qm9 import QM9Properties
from models.egnn import EGNN
from tqdm import tqdm
import os
import glob
import wandb



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def regression(loaders, metrics, model, args, wandb_logger):

    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    train_loader = loaders[0]
    valid_loader = loaders[1]
    test_loader = loaders[2]

    mean = metrics[0]
    mad = metrics[1]
    mad = mad.to(device)
    mean = mean.to(device)

    metric = torchmetrics.MeanAbsoluteError().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-1, verbose=True)

    def get_target(batch):
        return batch.y[:, 1]


    epoch_train_loss = []
    epoch_train_mae = []
    epoch_valid_mae = []

    for idx_epoch, epoch in enumerate(range(args.epochs)):

        # train loop

        model.train()
        epoch_loss = 0
        train_mae = 0

        for idx_train, batch in enumerate(train_loader):
            batch.to(device)

            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch).squeeze()
            target = get_target(batch).to(device)
            
            loss = F.l1_loss(pred, (target-mean)/mad)
            mae = metric(pred*mad + mean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            epoch_loss = epoch_loss + loss.item()
            train_mae = train_mae + mae.item()


        epoch_loss = epoch_loss/(idx_train+1)
        train_mae = train_mae/(idx_train+1)
        
        epoch_train_loss.append(epoch_loss)
        epoch_train_mae.append(train_mae)
        

        
        model.eval()
        valid_mae = 0
        with torch.no_grad():
            for index_valid, batch in enumerate(valid_loader):
                batch.to(device)
                
                pred = model(batch.x, batch.pos, batch.edge_index, batch.batch).squeeze()
                target = get_target(batch).to(device)
                
                mae = metric(pred*mad + mean, target)
                valid_mae = valid_mae + mae

            valid_mae = valid_mae/(index_valid+1)
            epoch_valid_mae.append(valid_mae)
            
    
        wandb_logger.log({"Train Loss":epoch_loss, "Train MAE": train_mae, "Valid MAE": valid_mae, "Learning Rate": optimizer.param_groups[-1]['lr']})
        scheduler.step() # after every epoch perform lr scheduling
        
        filename = "./checkpoints/model_epoch_" + str(epoch) +".pt"
        torch.save(model.state_dict(), filename)
        print("model saved")
    

        

        # Remove the oldest weights file if more than 3 checkpoints have been saved
        checkpoints = glob.glob('./checkpoints/*')
        if len(checkpoints) > 50:
            oldest_checkpoint = sorted(checkpoints, key=os.path.getctime)[0]
            os.remove(oldest_checkpoint)
    
            


    # test at last epoch   

    model.eval()
    test_mae = 0
    with torch.no_grad():
        for idx_test, batch in enumerate(test_loader):
            batch.to(device)
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch).squeeze()
            target = get_target(batch).to(device)

            mae = metric(pred*mad + mean, target)
            test_mae = test_mae + mae

        test_mae = test_mae/(idx_test+1)
    print("test ended")
    wandb_logger.log({"Test Mae: ", test_mae})


    
    wandb.finish()





    

        
















