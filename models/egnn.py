import torch
import torch.nn as nn

import torch_geometric as tg
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
import torch_geometric.utils as utils
from torch_geometric.utils import erdos_renyi_graph, to_dense_batch

class EGNNLayer(MessagePassing):
    """ E(n)-equivariant Message Passing Layer """
    def __init__(self, node_features, edge_features, hidden_features, out_features, dim, aggr, act):
        super().__init__(aggr=aggr)
        self.dim = dim

        self.message_net = nn.Sequential(nn.Linear(2 * node_features + edge_features, hidden_features),
                                         act(),
                                         nn.Linear(hidden_features, hidden_features),
                                         act())
    
        self.update_net = nn.Sequential(nn.Linear(node_features + hidden_features, hidden_features),
                                        act(),
                                        nn.Linear(hidden_features, out_features))
        
        # self.pos_net = nn.Sequential(nn.Linear(hidden_features, hidden_features), # removed bcs qm9 does not require position update
        #                              act(),
        #                              nn.Linear(hidden_features, 1))
        # nn.init.xavier_uniform_(self.pos_net[-1].weight, gain=0.001)

        self.edge_net = nn.Sequential(nn.Linear(hidden_features, 1), nn.Sigmoid())
        
        

    

    def forward(self, x, edge_index, edge_attr=None):
       
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_i, x_j,  edge_attr):
        """ Create messages """
        input = [x_i, x_j] if edge_attr is None else [x_i, x_j, edge_attr]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)
        is_edge = self.edge_net(message) 
        message = message * is_edge 
    
        # pos_message = (pos_i - pos_j)*self.pos_net(message) # we do not update here the pos
        # message = torch.cat((message, pos_message), dim=-1)
        

        return message #m_ij

    def update(self, message, x):
        """ Update node features and positions """
        # node_message, pos_message = message[:, :-self.dim], message[:, -self.dim:] # we dont return pos_message in qm9
        # node_message = message[:, :-self.dim]
        node_message = message
        # Update node features
        input = torch.cat((x, node_message), dim=-1)
        x += self.update_net(input)
        # Update positions
        # pos += pos_message # we do not update the positions anymore
        return x

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
            new_layer = EGNNLayer(hidden_features, edge_features, hidden_features, hidden_features, dim, aggr, act)
            layers.append(new_layer)
        self.layers = nn.ModuleList(layers)

        self.pooler = pool

        #TODO: Original Paper uses only one Linear layer
        self.head = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                  act(),
                                  nn.Linear(hidden_features, out_features))
        self.node_dec = nn.Sequential(nn.Linear(hidden_features, hidden_features), act(), nn.Linear(hidden_features, hidden_features))

    def forward(self, x, pos, edge_index, batch):

        """
        We are connecting all the graph nodes when the dataloaders are made.

        num_nodes = x.shape[0]
        edge_index = [] # We dont care about the original edge_index
        #For each batch(molecule) we fully connect its nodes and create a separate edge_index
        for b in range(batch.max().item() + 1): # for each molecule
            mask = (batch == b).view(-1, 1)  # check whether it is that specific molecule, mask: tensor (num_nodes, 1), true if node is from molecule b
            indices = torch.arange(num_nodes).view(-1, 1) 
            indices = indices[mask.expand_as(indices)].view(-1) # indexes of the node of the current molecule
            edges = torch.cartesian_prod(indices, indices)
            edges = edges[edges[:, 0] != edges[:, 1]]  # Remove self-edges 
            edge_index.append(edges)
        edge_index = torch.cat(edge_index, dim=0).t().contiguous() # We join the separate edge_indexes for each molecule 



        """
        # Compute edge distances
        dist = torch.sum((pos[edge_index[1]] - pos[edge_index[0]]).pow(2), dim=-1, keepdim=True).sqrt()
        edge_attr = dist

        # Feedforward through EGNNLayers
        x = self.embedder(x)
        for layer in self.layers:
            # x, pos = layer(x, pos, edge_index, edge_attr) # we do not return the pos anymore
            x = layer(x, edge_index, edge_attr)
        


        x = self.node_dec(x)
        if self.pooler:
            x = self.pooler(x, batch)

        x = self.head(x)
        return x
    


        """
        # ORIGINAL IMPLEMENTATION--> batch dist is changing 
        x = self.embedder(x)
        edge_index = erdos_renyi_graph(x.shape[0], 1.0).to("cuda") # added this one to try ->  1.0 specifies the probability of connecting any two nodes with an edge.
        # the edge index should be of size  [2, num_nodes * (num_nodes - 1)]-> need to compare with the satoras code to see the dimension of this
        dist = torch.sum((pos[edge_index[1]] - pos[edge_index[0]]).pow(2), dim=-1, keepdim=True).sqrt()
        edge_attr = dist

        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, dist) 
            

            # Update graph. If it is set to None, then a fully connected graph is implied.
            
            # if self.radius:  
            #     edge_index = tg.nn.radius_graph(pos, self.radius, batch) function does not even work when used.
            #     dist = torch.sum((pos[edge_index[1]] - pos[edge_index[0]]).pow(2), dim=-1, keepdim=True).sqrt()
        
        if self.pooler:
            x = self.pooler(x, batch)
        
        x = self.head(x)
        """

        """
        Something that i tried regarding the fully connected graph. you can ignore. just for reference


        x = self.embedder(x)

        dist = torch.sum((pos[edge_index[1]] - pos[edge_index[0]]).pow(2), dim=-1, keepdim=True).sqrt()
        edge_attr = dist

        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, dist) 
            
        # Update graph using radius or fully connected graph.
        if self.radius:
            edge_index = tg.nn.radius_graph(pos, self.radius, batch)
            dist = torch.sum((pos[edge_index[1]] - pos[edge_index[0]]).pow(2), dim=-1, keepdim=True).sqrt()
            edge_attr = dist
        else:
            num_nodes = pos.shape[0]
            row = torch.arange(num_nodes, device=pos.device)
            col = torch.arange(num_nodes, device=pos.device)
            row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
            col = col.repeat(num_nodes)
            edge_index = torch.stack([row, col], dim=0)
            dist = torch.sum((pos[edge_index[1]] - pos[edge_index[0]]).pow(2), dim=-1, keepdim=True).sqrt()
            edge_attr = dist[edge_index[1]]

        if self.pooler:
            x = self.pooler(x, batch)
        
        x = self.head(x)
        """

        return x