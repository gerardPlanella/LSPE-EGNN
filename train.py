import time
import torch

from tqdm import tqdm

def evaluate(model, loader, criterion, device, mean, mad):
    mae = 0.0
    model.eval()
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        target = torch.squeeze(batch.y[:, 1])
        pred = model(batch)
        loss = criterion(pred * mad + mean, target)
        mae += loss.item()

    return mae / len(loader.dataset)

def train(model, loader, criterion,
          optimizer, device,
          mean, mad):
    mae = 0.0
    model.train()
    for _, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        target = torch.squeeze(batch.y[:, 1]).to(device)

        # Perform forward pass
        pred = model(batch)

        # Calculate train loss
        loss = criterion(pred, (target - mean) / mad)
        mae += criterion(pred * mad + mean, target).item()

        # Delete info on previous gradients
        optimizer.zero_grad()

        # Propagate & optimizer step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return mae / len(loader.dataset)
