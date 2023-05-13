import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool

class EGNNLayer(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(4 * num_hidden + 1, num_hidden), nn.SiLU(),
                                         nn.Linear(num_hidden, num_hidden), nn.SiLU())
        self.message_mlp_pos = nn.Sequential(nn.Linear(2 * num_hidden + 1, num_hidden), nn.Tanh(),
                                             nn.Linear(num_hidden, num_hidden), nn.Tanh())
        self.update_mlp = nn.Sequential(nn.Linear(3 * num_hidden, num_hidden), nn.SiLU(),
                                        nn.Linear(num_hidden, num_hidden))
        self.edge_net = nn.Sequential(nn.Linear(num_hidden, 1), nn.Sigmoid())
        self.update_pos_net = nn.Sequential(nn.Linear(2 * num_hidden, num_hidden), nn.Tanh(),
                                            nn.Linear(num_hidden, num_hidden), nn.Tanh())

    # Old forward variant, without MPS support
    # def forward(self, x, pos, edge_index, pe):
    #     send, rec = edge_index
    #     dist = torch.linalg.norm(pos[send] - pos[rec], dim=1).unsqueeze(1)
    #     state = torch.cat((torch.cat([x[send], pe[send]], dim=-1), torch.cat([x[rec], pe[rec]], dim=-1), dist), dim=1)
    #     state_pe = torch.cat([pe[send], pe[rec], dist], dim=1)
    #
    #     message = self.message_mlp(state)
    #     message_pos = self.message_mlp_pos(state_pe)
    #     # message = self.edge_net(message_pre) * message_pre
    #     aggr = scatter_add(message, rec, dim=0)
    #     aggr_pos = scatter_add(message_pos, rec, dim=0)
    #     update = self.update_mlp(torch.cat((x, pe, aggr), dim=1))
    #     update_pe = self.update_pos_net(torch.cat([pe, aggr_pos], dim=1))
    #     return update, update_pe

    def forward(self, x, pos, edge_index, pe):
        send, rec = edge_index
        dist = torch.norm(pos[send] - pos[rec], dim=1)
        state = torch.cat((torch.cat([x[send], pe[send]], dim=-1),
                           torch.cat([x[rec], pe[rec]], dim=-1),
                           dist.unsqueeze(1)), dim=1)
        state_pe = torch.cat([pe[send], pe[rec], dist], dim=1)

        message = self.message_mlp(state)
        message_pos = self.message_mlp_pos(state_pe)

        # message = self.edge_net(message_pre) * message_pre
        aggr = torch.zeros((x.size(0), message.size(1)), device=x.device)
        aggr_pos = torch.zeros((x.size(0), message_pos.size(1)), device=x.device)

        aggr = aggr.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message)
        aggr_pos = aggr_pos.scatter_add(0, rec.unsqueeze(1).expand(send.size(0), message.size(1)), message_pos)

        update = self.update_mlp(torch.cat((x, pe, aggr), dim=1))
        update_pe = self.update_pos_net(torch.cat([pe, aggr_pos], dim=1))

        return update, update_pe


class EGNN(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, num_layers):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(num_in, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.embed_pe = nn.Sequential(nn.Linear(15, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.layers = nn.ModuleList([EGNNLayer(num_hidden) for _ in range(num_layers)])
        self.pre_readout = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.SiLU(),
                                         nn.Linear(num_hidden, num_hidden))
        self.readout = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_out))

    def forward(self, data):
        x, pos, edge_index, batch, rw = data.x, data.pos, data.edge_index, data.batch, data.random_walk_pe
        # x = torch.cat([x, rw], dim = -1)
        x = self.embed(x)
        pe = self.embed_pe(rw)

        for layer in self.layers:
            # x = x + layer(x, pos, edge_index)
            out, pe_out = layer(x, pos, edge_index, pe)
            x = x + out
            pe = pe_out + pe

        x = self.pre_readout(x)
        x = global_add_pool(x, batch)
        out = self.readout(x)

        return torch.squeeze(out)
