import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from .utils import get_pe_attribute

class MPNNLayer(nn.Module):
    """Vanilla MPNN layer"""
    def __init__(self, num_hidden, **kwargs):
        super().__init__()
        self.both_states = kwargs['both_states']
        self.num_hidden = num_hidden
        self.message_mlp = nn.Sequential(
            nn.Linear(num_hidden, num_hidden), nn.SiLU(),
            nn.Linear(num_hidden, num_hidden), nn.SiLU())

    def forward(self, x, _, edge_index):
        send, rec = edge_index
        state = x[send]
        if self.both_states:
            state = torch.cat([state, x[rec]], dim=1)
        message = self.message_mlp(state)
        aggr = torch.zeros((x.size(0), message.size(1)), device=x.device)
        aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message)
        update = x + aggr
        return update


class MPNNLSPELayer(nn.Module):
    """MPNN layer augmented with LSPE"""

    def __init__(self, num_hidden, **kwargs):
        super().__init__()
        self.include_dist = kwargs['include_dist']
        params_dist = 1 if self.include_dist else 0
        self.message_mlp = nn.Sequential(
            nn.Linear(4 * num_hidden + params_dist, num_hidden), nn.SiLU(),
            nn.Linear(num_hidden, num_hidden), nn.SiLU())
        self.message_mlp_pos = nn.Sequential(
            nn.Linear(2 * num_hidden + 1, num_hidden), nn.Tanh(),
            nn.Linear(num_hidden, num_hidden), nn.Tanh())

    def forward(self, x, pos, edge_index, pe):
        send, rec = edge_index

        state = torch.cat((torch.cat([x[send], pe[send]], dim=-1),
                           torch.cat([x[rec], pe[rec]], dim=-1)), dim=1)
        state_pe = torch.cat([pe[send], pe[rec]], dim=1)

        if self.include_dist:
            dist = torch.norm(pos[send] - pos[rec], dim=1).unsqueeze(1)
            state = torch.cat((state, dist), dim=1)
            state_pe = torch.cat((state, dist), dim=1)

        message = self.message_mlp(state)
        message_pos = self.message_mlp_pos(state_pe)

        aggr = torch.zeros((x.size(0), message.size(1)), device=x.device)
        aggr_pos = torch.zeros((x.size(0), message_pos.size(1)), device=x.device)

        aggr = aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message)
        aggr_pos = aggr_pos.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message_pos)

        update = x + aggr
        update_pe = pe + aggr_pos

        return update, update_pe


class MPNN(nn.Module):

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

        self.include_dist = kwargs['include_dist']
        self.both_states = kwargs['both_states']

        # Pre-condition in case we are using LSPE
        assert not (self.pe == 'nope' and self.lspe), "LSPE has to have initialized PE."

        # Initialization of embedder for the input features (node features + (optional) PE dim)
        self.embed = nn.Sequential(
            nn.Linear(self.in_channels + self.pe_dim, self.hidden_channels))

        # If the LSPE framework is used, add another embedder
        if self.lspe:
            self.embed_pe = nn.Sequential(
                nn.Linear(self.pe_dim, self.hidden_channels))

        # Initialization of hidden MPNN with (optional) LSPE hidden layers
        layer = MPNNLSPELayer if self.lspe else MPNNLayer
        self.layers = nn.ModuleList([
            layer(self.hidden_channels, **kwargs) for _ in range(num_layers)])

        # Readout networks
        self.pre_readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.out_channels))

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch

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
