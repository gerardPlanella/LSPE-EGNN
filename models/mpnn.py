import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add
from .utils import get_pe_attribute

class MPNNLayer(nn.Module):
    """Vanilla MPNN layer"""

    def __init__(self, num_hidden, **kwargs):
        super().__init__()
        self.num_hidden = num_hidden
        self.reduced = kwargs['reduced']
        self.include_dist = kwargs['include_dist']
        params_dist = 1 if self.include_dist else 0
        params_reduced_factor = 2 if self.reduced else 1

        # Message network: phi_m
        self.message_mlp = nn.Sequential(
            nn.Linear(2//params_reduced_factor * num_hidden + params_dist, num_hidden), nn.SiLU(),
            nn.Linear(num_hidden, num_hidden), nn.SiLU())

        # Update network: phi_h
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * num_hidden, num_hidden), nn.SiLU(),
            nn.Linear(num_hidden, num_hidden))

    def forward(self, x, pos, edge_index):
        send, rec = edge_index
        state = x[send]

        if not self.reduced:
            state = torch.cat([state, x[rec]], dim=1)

        if self.include_dist:
            dist = torch.linalg.norm(pos[send] - pos[rec], dim=1).unsqueeze(1)
            state = torch.cat([state, dist], dim=1)

        # Pass the state through the message net
        message = self.message_mlp(state)

        # Aggregate messages from neighbours by summing
        # aggr = torch.zeros((x.size(0), message.size(1)), device=x.device)
        # aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message)
        aggr = scatter_add(message, rec, dim=0)

        # Pass the new state through the update network alongside x
        update = self.update_mlp(torch.cat([x, aggr], dim=1))
        return update


class MPNNLSPELayer(nn.Module):
    """MPNN layer augmented with LSPE"""

    def __init__(self, num_hidden, **kwargs):
        super().__init__()
        self.include_dist = kwargs['include_dist']
        self.reduced = kwargs['reduced']
        self.update_with_pe = kwargs['update_with_pe']
        params_dist = 1 if self.include_dist else 0
        params_reduced_factor = 2 if self.reduced else 1
        params_update_with_pe_factor = 3 if self.update_with_pe else 2

        # Message network: phi_m
        self.message_mlp = nn.Sequential(
            nn.Linear(4//params_reduced_factor * num_hidden + params_dist, num_hidden), nn.SiLU(),
            nn.Linear(num_hidden, num_hidden), nn.SiLU())

        # PE LSPE network
        self.pos_mlp = nn.Sequential(
            nn.Linear(2//params_reduced_factor * num_hidden + params_dist, num_hidden), nn.Tanh(),
            nn.Linear(num_hidden, num_hidden), nn.Tanh())

        # Update network: phi_h
        self.update_mlp = nn.Sequential(
            nn.Linear(params_update_with_pe_factor * num_hidden, num_hidden), nn.SiLU(),
            nn.Linear(num_hidden, num_hidden))

        # PE update network: phi_p
        self.update_mlp_pos = nn.Sequential(
            nn.Linear(2 * num_hidden, num_hidden), nn.Tanh(),
            nn.Linear(num_hidden, num_hidden), nn.Tanh())

    def forward(self, x, pos, edge_index, pe):
        send, rec = edge_index
        state = torch.cat([x[send], pe[send]], dim=-1)
        pe_state = pe[send]

        if not self.reduced:
            state = torch.cat([state, torch.cat([x[rec], pe[rec]], dim=-1)], dim=1)
            pe_state = torch.cat([pe_state, pe[rec]], dim=1)

        if self.include_dist:
            dist = torch.linalg.norm(pos[send] - pos[rec], dim=1).unsqueeze(1)
            state = torch.cat([state, dist], dim=1)
            pe_state = torch.cat([pe_state, dist], dim=1)

        # Pass the state through the message net
        message = self.message_mlp(state)
        pos = self.pos_mlp(pe_state)

        # Aggregate state messages from neighbours by summing
        # aggr = torch.zeros((x.size(0), message.size(1)), device=x.device)
        # aggr = aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message)
        aggr = scatter_add(message, rec, dim=0)

        # Pass the new state through the update network alongside x
        update_state = [x, pe, aggr] if self.update_with_pe else [x, aggr]
        update = self.update_mlp(torch.cat(update_state, dim=1))

        # Aggregate pos from neighbourhood by summing
        # pos_aggr = torch.zeros((x.size(0), pos.size(1)), device=x.device)
        # pos_aggr = pos_aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), pos.size(1)), pos)
        pos_aggr = scatter_add(pos, rec, dim=0)

        # Pass the new pos state through the update network alongside pe
        update_pe = self.update_mlp_pos(torch.cat([pe, pos_aggr], dim=1))
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
                x += layer(x, pos, edge_index)

        x = self.pre_readout(x)
        x = global_add_pool(x, batch)
        out = self.readout(x)

        return torch.squeeze(out)
