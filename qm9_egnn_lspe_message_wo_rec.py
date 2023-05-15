import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import copy
import wandb
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool
from torch_geometric.datasets import QM9
from torch_geometric.transforms import RadiusGraph, AddRandomWalkPE, Compose


class EGNNLayer(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(2 * num_hidden + 1, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden), nn.SiLU())
        self.message_mlp_pos = nn.Sequential(nn.Linear(2 * num_hidden + 1, num_hidden), nn.Tanh(), nn.Linear(num_hidden, num_hidden), nn.Tanh())
        self.update_mlp = nn.Sequential(nn.Linear(3 * num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.edge_net = nn.Sequential(nn.Linear(num_hidden, 1), nn.Sigmoid())
        self.update_pos_net = nn.Sequential(nn.Linear(2*num_hidden, num_hidden), nn.Tanh(), nn.Linear(num_hidden, num_hidden), nn.Tanh())

    def forward(self, x, pos, edge_index, pe):
        send, rec = edge_index
        dist = torch.linalg.norm(pos[send] - pos[rec], dim=1).unsqueeze(1)
        state = torch.cat((torch.cat([x[rec], pe[rec]], dim = -1), dist), dim=1)
        state_pe = torch.cat([pe[send], pe[rec], dist], dim = 1)

        message = self.message_mlp(state)
        message_pos = self.message_mlp_pos(state_pe)

        # message = self.edge_net(message_pre) * message_pre
        aggr = scatter_add(message, rec, dim=0)
        aggr_pos = scatter_add(message_pos, rec, dim = 0)

        update = self.update_mlp(torch.cat((x, pe, aggr), dim = 1))
        update_pe = self.update_pos_net(torch.cat([pe, aggr_pos], dim = 1))
        
        return update, update_pe


class EGNN(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, num_layers):
        super().__init__()
        self.embed = nn.Linear(num_in, num_hidden)
        self.embed_pe = nn.Linear(24, num_hidden)
        self.layers = nn.ModuleList([EGNNLayer(num_hidden) for _ in range(num_layers)])
        self.pre_readout = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
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


if __name__ == '__main__':
    wandb.init(project=f"DL2-EGNN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transform = RadiusGraph(r=1e6)
    t_compose = Compose([AddRandomWalkPE(walk_length = 24)])
    dataset = QM9('./data_PE24_no_connected/PE/data_PE24_no_connected', pre_transform = t_compose)
    epochs = 1000

    n_train, n_test = 100000, 110000
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:n_test]
    val_dataset = dataset[n_test:]

    print("Total number of edges: ", train_dataset.data.edge_index.shape[1] + val_dataset.data.edge_index.shape[1] + test_dataset.data.edge_index.shape[1])


    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)
    

    # graph.y[:, 1] for alpha
    values = [torch.squeeze(graph.y[:, 1]) for graph in train_loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    mean, mad = mean.to(device), mad.to(device)

    model = EGNN(
        num_in=11,
        num_hidden=128,
        num_out=1,
        num_layers=7
    ).to(device)

    criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-16)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_train_mae, best_val_mae, best_model = float('inf'), float('inf'), None

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')

    for _ in tqdm(range(epochs)):
        epoch_mae_train, epoch_mae_val = 0, 0

        model.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = model(batch)
            target = torch.squeeze(batch.y[:, 1])  # batch.y[1] for alpha
            loss = criterion(pred, (target - mean) / mad)
            mae = criterion(pred * mad + mean, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_mae_train += mae.item()

        model.eval()
        for _, batch in enumerate(val_loader):
            batch = batch.to(device)
            target = torch.squeeze(batch.y[:, 1])  # batch.y[1] for alpha
            pred = model(batch)

            mae = criterion(pred * mad + mean, target)

            epoch_mae_val += mae.item()

        epoch_mae_train /= len(train_loader.dataset)
        epoch_mae_val /= len(val_loader.dataset)

        if epoch_mae_val < best_val_mae:
            best_val_mae = epoch_mae_val
            best_model = copy.deepcopy(model)

        scheduler.step()

        wandb.log({
            'Train MAE': epoch_mae_train,
            'Validation MAE': epoch_mae_val
        })

    test_mae = 0
    best_model.eval()
    for _, batch in enumerate(test_loader):
        batch = batch.to(device)
        target = torch.squeeze(batch.y[:, 1])  # batch.y[:, 1] for alpha

        pred = best_model(batch)
        mae = criterion(pred * mad + mean, target)
        test_mae += mae.item()

    test_mae /= len(test_loader.dataset)
    print(f'Test MAE: {test_mae}')

    wandb.log({
        'Test MAE': test_mae,
    })
