import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from dataset.qm9 import QM9Properties
from models.egnn import EGNN

class QM9Regressor(pl.LightningModule):
    def __init__(self, model, target:QM9Properties, lr, weight_decay, epochs, mean=0, mad=1):
        super().__init__()
        self.model = model
        assert target in QM9Properties
        self.target = target
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()


        self.mean = mean
        self.mad = mad

        self.save_hyperparameters(ignore=['model'])

    def forward(self, graph):
        if isinstance(self.model, EGNN):
            # Don't add distance, as this is done internally
            pred = self.model(graph.x, graph.pos, graph.edge_index, graph.batch)
                    
        return pred

    def get_target(self, graph):
        return graph.y[:, self.target.value]

    def training_step(self, graph):
        pred = self(graph).squeeze()
        y = self.get_target(graph)
        loss = F.l1_loss(pred, (y - self.mean)/self.mad)
        self.train_metric(pred*self.mad + self.mean, y)
        return loss

    def on_train_epoch_end(self):
        self.log("train_MAE", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pred = self(graph).squeeze()
        y = self.get_target(graph)
        self.valid_metric(pred*self.mad + self.mean, y)

    def test_step(self, graph, batch_idx):
        pred = self(graph).squeeze()
        y = self.get_target(graph)
        self.test_metric(pred*self.mad + self.mean, y)

    def on_test_epoch_end(self):
        self.log("test_MAE", self.test_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("valid_MAE", self.valid_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, amsgrad=True, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.epochs-1, verbose=True
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]