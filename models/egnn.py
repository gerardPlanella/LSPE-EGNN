import torch
import torch.nn as nn

from .utils import get_pe_attribute
from torch_geometric.nn import global_add_pool


class EGNNLayer(nn.Module):
    """Standard version of the EGNN layer"""

    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_channels + 1, self.hidden_channels),
            nn.SiLU(), nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU())
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))
        self.edge_net = nn.Sequential(
            nn.Linear(self.hidden_channels, 1), nn.Sigmoid())

    def forward(self, x, pos, edge_index):
        send, rec = edge_index
        dist = torch.norm(pos[send] - pos[rec], dim=1)
        state = torch.cat((x[send], x[rec], dist.unsqueeze(1)), dim=1)
        message = self.message_mlp(state)
        aggr = torch.zeros((x.size(0), message.size(1)), device=x.device)
        aggr = aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message)
        update = self.update_mlp(torch.cat((x, aggr), dim=1))
        return update


class EGNNLSPELayer(nn.Module):
    """EGNN layer augmented with LSPE"""

    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.message_mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_channels + 1, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU())
        self.message_pos_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_channels + 1, self.hidden_channels), nn.Tanh(),
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.Tanh())
        self.update_mlp = nn.Sequential(
            nn.Linear(3 * self.hidden_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))
        self.update_pos_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_channels, self.hidden_channels), nn.Tanh(),
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.Tanh())
        self.edge_net = nn.Sequential(
            nn.Linear(self.hidden_channels, 1), nn.Sigmoid())

    def forward(self, x, pos, edge_index, pe):
        send, rec = edge_index
        dist = torch.norm(pos[send] - pos[rec], dim=1)
        state = torch.cat((torch.cat([x[send], pe[send]], dim=-1),
                           torch.cat([x[rec], pe[rec]], dim=-1),
                           dist.unsqueeze(1)), dim=1)
        state_pe = torch.cat([pe[send], pe[rec], dist], dim=1)

        message = self.message_mlp(state)
        message_pos = self.message_pos_mlp(state_pe)

        aggr = torch.zeros((x.size(0), message.size(1)), device=x.device)
        aggr_pos = torch.zeros((x.size(0), message_pos.size(1)), device=x.device)

        aggr = aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message)
        aggr_pos = aggr_pos.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message_pos)

        update = self.update_mlp(torch.cat((x, pe, aggr), dim=1))
        update_pe = self.update_pos_mlp(torch.cat([pe, aggr_pos], dim=1))

        return update, update_pe


class EGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels,
                 pe='rw', pe_dim=24, lspe=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.pe = pe
        self.pe_dim = pe_dim if pe != 'nope' else 0
        self.lspe = lspe

        # Pre-condition in case we are using LSPE
        assert self.pe != 'nope' and self.lspe, "LSPE has to have initialized PE."

        # Initialization of embedder for the input features (node features + (optional) PE dim)
        self.embed = nn.Sequential(
            nn.Linear(self.in_channels + self.pe_dim, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))

        # If the LSPE framework is used, add another embedder
        if self.lspe:
            self.embed_pe = nn.Sequential(
                nn.Linear(self.pe_dim, self.hidden_channels), nn.SiLU(),  # todo prev, in channels was 15?
                nn.Linear(self.hidden_channels, self.hidden_channels))

        # Initialization of hidden EGNN with (optional) LSPE hidden layers
        layer = EGNNLSPELayer if self.lspe else EGNNLayer
        self.layers = nn.ModuleList([
            layer(self.hidden_channels) for _ in range(self.num_layers)])

        # Readout networks
        self.pre_readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.out_channels))

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch

        # In case we have LSPE or solely PE init, initialize PE
        if self.pe != 'nope':
            pe = getattr(data, get_pe_attribute(self.pe))
            x = torch.cat([x, pe], dim=-1)

            # In the case of LSPE, pass PE through embedder
            pe = self.embed_pe(pe) if self.lspe else pe

        x = self.embed(x)

        for layer in self.layers:
            if self.lspe:
                out, pe_out = layer(x, pos, edge_index, pe)
                x += out
                pe += pe_out
            else:
                x = x + layer(x, pos, edge_index)

        x = self.pre_readout(x)
        x = global_add_pool(x, batch)
        out = self.readout(x)
        return torch.squeeze(out)
