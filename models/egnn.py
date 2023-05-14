import torch
import torch.nn as nn

from .utils import get_pe_attribute
from torch_geometric.nn import global_add_pool

class EGNNLayer(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * num_hidden + 1, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU())
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden))
        self.edge_net = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Sigmoid())

    # Old forward variant, without MPS support
    # def forward(self, x, pos, edge_index):
    #     send, rec = edge_index
    #     state = torch.cat((x[send], x[rec], torch.linalg.norm(pos[send] - pos[rec], dim=1).unsqueeze(1)), dim=1)
    #     message = self.message_mlp(state)
    #     # message = self.edge_net(message_pre) * message_pre
    #     aggr = scatter_add(message, rec, dim=0)
    #     update = self.update_mlp(torch.cat((x, aggr), dim=1))
    #     return update

    def forward(self, x, pos, edge_index):
        """New forward method with support for mps"""
        send, rec = edge_index
        dist = torch.norm(pos[send] - pos[rec], dim=1)
        state = torch.cat((x[send], x[rec], dist.unsqueeze(1)), dim=1)
        message = self.message_mlp(state)
        aggr = torch.zeros((x.size(0), message.size(1)), device=x.device)
        aggr = aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message)
        update = self.update_mlp(torch.cat((x, aggr), dim=1))
        return update


class EGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, pe='rw', pe_dim=24, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.pe = pe
        self.pe_dim = pe_dim if pe != 'nope' else 0
        self.embed = nn.Sequential(
            nn.Linear(self.in_channels + self.pe_dim, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))
        self.layers = nn.ModuleList([
            EGNNLayer(self.hidden_channels) for _ in range(self.num_layers)])
        self.pre_readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.out_channels))

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        if self.pe != 'nope':
            pe = getattr(data, get_pe_attribute(self.pe))
            x = torch.cat([x, pe], dim=-1)
        x = self.embed(x)

        for layer in self.layers:
            x = x + layer(x, pos, edge_index)

        x = self.pre_readout(x)
        x = global_add_pool(x, batch)
        out = self.readout(x)
        return torch.squeeze(out)
