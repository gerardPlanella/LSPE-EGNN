import torch
from dataset.qm9 import QM9Properties

def get_mean_and_mad(train_dataset, property):
    
    if isinstance(property, QM9Properties):
        values = []
        n = 0
        for batch in train_dataset:
            values.extend(batch.y[:, property.value])
            
        mean = torch.mean(torch.tensor(values))
        mad = torch.mean(torch.abs(torch.tensor(values) - mean))
        return mean, mad
    else:
        raise NotImplementedError
