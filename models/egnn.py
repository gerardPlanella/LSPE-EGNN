import torch
import torch.nn as nn

import torch_geometric as tg
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool

class EGNNLayer(MessagePassing):
    """ E(n)-equivariant Message Passing Layer """
    def __init__(self, node_features, edge_features, hidden_features, out_features, dim, aggr, act):
        super().__init__(aggr=aggr)
        self.dim = dim

        self.message_net = nn.Sequential(nn.Linear(2 * node_features + edge_features, hidden_features),
                                         act(),
                                         nn.Linear(hidden_features, hidden_features))
    
        self.update_net = nn.Sequential(nn.Linear(node_features + hidden_features, hidden_features),
                                        act(),
                                        nn.Linear(hidden_features, out_features))
        
        self.pos_net = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                     act(),
                                     nn.Linear(hidden_features, 1))

        self.edge_net = nn.Sequential(nn.Linear(hidden_features, 1), nn.Sigmoid())
        
        nn.init.xavier_uniform_(self.pos_net[-1].weight, gain=0.001)

    

    def forward(self, x, pos, edge_index, edge_attr=None):

        x = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
        return x

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        """ Create messages """
        input = [x_i, x_j] if edge_attr is None else [x_i, x_j, edge_attr]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)
        is_edge = self.edge_net(message) 
        message = message * is_edge # should it be before or after the pos net
    
        pos_message = (pos_i - pos_j)*self.pos_net(message)
        message = torch.cat((message, pos_message), dim=-1)
        
        return message #m_ij

    def update(self, message, x, pos):
        """ Update node features and positions """
        node_message, pos_message = message[:, :-self.dim], message[:, -self.dim:]
        # Update node features
        input = torch.cat((x, node_message), dim=-1)
        update = self.update_net(input)
        # Update positions
        pos += pos_message
        return update, pos

class EGNN(nn.Module):
    """ E(n)-equivariant Message Passing Network """
    def __init__(self, node_features, hidden_features, out_features, num_layers, dim, radius, aggr="mean", act=nn.ReLU, pool=global_add_pool):
        super().__init__()
        edge_features = 1
        self.dim = dim
        self.radius = radius

        self.embedder = nn.Sequential(nn.Linear(node_features, hidden_features),
                                      act(),
                                      nn.Linear(hidden_features, hidden_features))
        
       
    
        layers = []
        for i in range(num_layers):
            layers.append(EGNNLayer(hidden_features, edge_features, hidden_features, hidden_features, dim, aggr, act))
        self.layers = nn.ModuleList(layers)

        self.pooler = pool

        #TODO: Original Paper uses only one Linear layer
        self.head = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                  act(),
                                  nn.Linear(hidden_features, out_features))

    def forward(self, x, pos, edge_index, batch):
        x = self.embedder(x)

        dist = torch.sum((pos[edge_index[1]] - pos[edge_index[0]]).pow(2), dim=-1, keepdim=True).sqrt()
        edge_attr = dist
        edge_index = tg.utils.dense_to_sparse(torch.ones((x.size(0), x.size(0))), edge_attr=None)[0] #to try
       
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, dist) 
            

            # Update graph. If it is set to None, then a fully connected graph is implied.
            if self.radius:
                edge_index = tg.nn.radius_graph(pos, self.radius, batch)
                dist = torch.sum((pos[edge_index[1]] - pos[edge_index[0]]).pow(2), dim=-1, keepdim=True).sqrt()
        
        if self.pooler:
            x = self.pooler(x, batch)
        
        x = self.head(x)
        return x