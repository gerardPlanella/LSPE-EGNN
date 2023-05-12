import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import time
import re
import numpy as np
import argparse

from tqdm import tqdm
from pprint import pprint
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from train import train, evaluate
from torch_geometric.transforms import RadiusGraph, AddRandomWalkPE, AddLaplacianEigenvectorPE, Compose

# Plotting via wandb
import wandb


def setup_gpu():
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.get_device_name(0))
        print("CUDA available. Setting device to CUDA:", device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS available. Setting device to MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU or MPS available. Setting device to CPU.")
    return device


def set_seed(seed):
    """Function for setting the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_options():
    parser = argparse.ArgumentParser("Model runner.")

    # General Training parameters
    parser.add_argument('--model', type=str, default='egnn', metavar='N',
                        help='Available models: egnn | egnn_lspe')
    # todo: add another parameter for pe, which contains the dim, leave dataset as qm9 | qm9_fc
    # todo: see how this affects the change in the ouput files
    # todo: create a new get_pe method
    parser.add_argument('--dataset', type=str, default='qm9_rw24', metavar='N',
                        help='qm9 | qm9_(fc)_(rw<dim>)_(lap<dim>)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='Random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                        help='Batch size for training (default: 96)')
    parser.add_argument('--learning_rate', type=float, default=5e-4, metavar='N',
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='clamp the output of the coords function if get too large')

    # Network specific parameters
    parser.add_argument('--num_in', type=int, default=11, metavar='N',
                        help='Input dimension of features (default: 11)')
    parser.add_argument('--num_hidden', type=int, default=128, metavar='N',
                        help='Hidden dimensions (default: 128)')
    parser.add_argument('--num_out', type=int, default=1, metavar='N',
                        help='Output dimensions (default: 1)')
    parser.add_argument('--num_layers', type=int, default=7, metavar='N',
                        help='Number of model layers (default: 7)')

    # todo: embedder linear layers
    args = parser.parse_args()
    return args

def split_qm9(dataset):
    n_train, n_test = 100000, 110000
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:n_test]
    val_dataset = dataset[n_test:]
    return train_dataset, val_dataset, test_dataset

def get_dataset(dataset_name, **kwargs):
    """Gets the corresponding QM9 dataset.
        Examples of dataset names: QM9_fc_rw42, QM9_lap42"""
    assert not (('rw' in dataset_name.lower()) and ('lap' in dataset_name.lower())), "Cannot have both RW and LAP as PE"
    transform = Compose([])
    if 'fc' in dataset_name.lower():
        transform.transforms.append(RadiusGraph(1e6))
    elif f'rw' in dataset_name.lower():
        try:
            pe_dim = int(re.findall(r'\d+', kwargs['dataset'])[-1])
            transform.transforms.append(AddRandomWalkPE(pe_dim))
        except TypeError:
            print('Please specify the Positional Encoding dimension for RW PE')
    elif f'lap' in dataset_name.lower():
        try:
            pe_dim = int(re.findall(r'\d+', kwargs['dataset'])[-1])
            transform.transforms.append(AddLaplacianEigenvectorPE(pe_dim))
        except TypeError:
            print('Please specify the Positional Encoding dimension for Laplacian Eigenvectors PE.')

    return QM9(f'./data/{dataset_name}', pre_transform=transform)

def get_model(model_name):
    if model_name == 'egnn':
        from qm9_egnn import EGNN
        return EGNN
    elif model_name == 'egnn_lspe':
        from qm9_egnn_lspe import EGNN
        return EGNN
    else:
        raise NotImplementedError(f'Model name {model_name} not recognized.')


def main(args):
    # Display run arguments
    pprint(args)

    # Set the hardware accelerator
    device = setup_gpu()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Get the dataset object
    dataset = get_dataset(args.dataset, **vars(args))

    # Split the dataset into train, val and test
    train_dataset, val_dataset, test_dataset = split_qm9(dataset)

    # Initialize the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Setting the WandB parameters
    numbers = re.findall(r'\d+', args.dataset)
    args.pe_dim = int(numbers[-1]) if len(numbers) > 0 else 0
    config = {
        **vars(args),
        'lspe': 'lspe' in args.model,
        'fc': 'fc' in args.model,
        # todo: change pe_dim to an argument for argparse
        'pe_dim': int(numbers[-1]) if len(numbers) > 0 else 0
    }
    wandb.init(project="dl2-project", entity="msc-ai", config=config, reinit=True,
               name=f'{args.model}_{args.dataset}')

    # Initialize the model
    net = get_model(args.model)
    model = net(
        num_in=args.num_in,
        num_hidden=args.num_hidden,
        num_out=args.num_out,
        num_layers=args.num_layers,
        pe_dim=args.pe_dim
    ).to(device)
    wandb.watch(model)

    # Declare the training criterion, optimizer and scheduler
    criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Number of parameters of the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')

    # Saving the best model instance based on validation MAE
    best_train_mae, best_val_mae, model_path = float('inf'), float('inf'), ""

    # Calculate the mean and mad of the dataset
    values = [torch.squeeze(graph.y[:, 1]) for graph in train_loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    mean, mad = mean.to(device), mad.to(device)

    print('Beginning training...')
    try:
        with tqdm(range(args.epochs)) as t:
            for epoch in t:
                t.set_description(f'Epoch {epoch}')
                start = time.time()
                epoch_train_mae = train(model, train_loader, criterion, optimizer, device, mean, mad)
                epoch_val_mae = evaluate(model, val_loader, criterion, device, mean, mad)

                wandb.log({'Train MAE': epoch_train_mae})
                wandb.log({'Validation MAE': epoch_val_mae})

                # Best model based on validation MAE
                if epoch_val_mae < best_val_mae:
                    best_val_mae = epoch_val_mae
                    wandb.run.summary["best_val_mae"] = best_val_mae
                    ckpt = {"state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_mae": best_val_mae,
                            "best_epoch": epoch}
                    model_path = f"saved_models/{args.model}" \
                                 f"_{args.dataset}" \
                                 f"_epochs-{args.epochs}" \
                                 f"_batch-{args.batch_size}" \
                                 f"_num_hidden-{args.num_hidden}" \
                                 f"_num_layers-{args.num_layers}.pt"
                    torch.save(ckpt, model_path)

                # Perform LR step
                scheduler.step()

                # Update the postfix of tqdm with every iteration
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_mae, val_loss=epoch_val_mae)

    except KeyboardInterrupt:
        print('Exiting training early because of keyboard interrupt.')

    # Load best model
    print('Loading best model...')
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["state_dict"])

    # Perform evaluation on test set
    print('Beginning evaluation...')
    test_mae = evaluate(model, test_loader, criterion, device, mean, mad)
    wandb.run.summary["test_mae"] = test_mae


if __name__ == '__main__':
    args = parse_options()
    main(args)

