import torch
import torch.nn as nn

from .utils import get_pe_attribute, deprecated
from torch_geometric.nn import global_add_pool

@deprecated
class EGNNLayer(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(4 * num_hidden + 1, num_hidden), nn.SiLU(),
            nn.Linear(num_hidden, num_hidden), nn.SiLU())
        self.message_mlp_pos = nn.Sequential(
            nn.Linear(2 * num_hidden + 1, num_hidden), nn.Tanh(),
            nn.Linear(num_hidden, num_hidden), nn.Tanh())
        self.update_mlp = nn.Sequential(
            nn.Linear(3 * num_hidden, num_hidden), nn.SiLU(),
            nn.Linear(num_hidden, num_hidden))
        self.edge_net = nn.Sequential(
            nn.Linear(num_hidden, 1), nn.Sigmoid())
        self.update_pos_net = nn.Sequential(
            nn.Linear(2 * num_hidden, num_hidden), nn.Tanh(),
            nn.Linear(num_hidden, num_hidden), nn.Tanh())

    def forward(self, x, pos, edge_index, pe):
        send, rec = edge_index
        dist = torch.norm(pos[send] - pos[rec], dim=1)
        state = torch.cat((torch.cat([x[send], pe[send]], dim=-1),
                           torch.cat([x[rec], pe[rec]], dim=-1),
                           dist.unsqueeze(1)), dim=1)
        state_pe = torch.cat([pe[send], pe[rec], dist], dim=1)

        message = self.message_mlp(state)
        message_pos = self.message_mlp_pos(state_pe)

        aggr = torch.zeros((x.size(0), message.size(1)), device=x.device)
        aggr_pos = torch.zeros((x.size(0), message_pos.size(1)), device=x.device)

        aggr = aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message)
        aggr_pos = aggr_pos.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message_pos)

        update = self.update_mlp(torch.cat((x, pe, aggr), dim=1))
        update_pe = self.update_pos_net(torch.cat([pe, aggr_pos], dim=1))

        return update, update_pe

@deprecated
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
            nn.SiLU(), nn.Linear(self.hidden_channels, self.hidden_channels))
        self.embed_pe = nn.Sequential(
            nn.Linear(15, self.hidden_channels),
            nn.SiLU(), nn.Linear(self.hidden_channels, self.hidden_channels))
        self.layers = nn.ModuleList(
            [EGNNLayer(self.hidden_channels) for _ in range(self.num_layers)])
        self.pre_readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(), nn.Linear(self.hidden_channels, self.hidden_channels))
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(), nn.Linear(self.hidden_channels, self.out_channels))

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        pe = getattr(data, get_pe_attribute(self.pe))
        x = torch.cat([x, pe], dim=-1)

        x = self.embed(x)
        pe = self.embed_pe(pe)

        for layer in self.layers:
            out, pe_out = layer(x, pos, edge_index, pe)
            x = x + out
            pe = pe_out + pe

        x = self.pre_readout(x)
        x = global_add_pool(x, batch)
        out = self.readout(x)

        return torch.squeeze(out)
