import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import copy
import time
import json
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

script_dir = os.path.dirname(__file__)

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

    # Config matters
    parser.add_argument('--config', type=str, default=None, metavar='S',
                        help='Config file for parsing arguments. ' 
                             'Command line arguments will be overriden.')
    parser.add_argument('--write_config_to', type=str, default=None, metavar='S',
                        help='Writes the current arguments as a json file for '
                             'config with the specified filename.')

    # General Training parameters
    parser.add_argument('--model', type=str, default='egnn', metavar='S',
                        help='Available models: egnn | egnn_lspe')
    parser.add_argument('--dataset', type=str, default='qm9', metavar='S',
                        help='Available datasets: qm9 | qm9_fc')
    parser.add_argument('--pe', type=str, default='rw', metavar='S',
                        help='Available PEs: nope | rw | lap')
    parser.add_argument('--pe_dim', type=int, default=24, metavar='N',
                        help='PE dimension')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, metavar='N',
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='clamp the output of the coords function if get too large')

    # Network specific parameters
    parser.add_argument('--in_channels', type=int, default=11, metavar='N',
                        help='Input dimension of features')
    parser.add_argument('--hidden_channels', type=int, default=128, metavar='N',
                        help='Hidden dimensions')
    parser.add_argument('--num_layers', type=int, default=7, metavar='N',
                        help='Number of model layers')
    parser.add_argument('--out_channels', type=int, default=1, metavar='N',
                        help='Output dimensions')

    args = parser.parse_args()

    if args.config is not None:
        config_dir_path = os.path.join(script_dir, 'config')
        with open(os.path.join(config_dir_path, args.config), 'r') as cf:
            parser.set_defaults(**json.load(cf))
            print(f'Successfully parsed the arguments from config/{args.config}')
        args = parser.parse_args()

    if args.write_config_to is not None:
        # If no config directory, make it
        config_dir_path = os.path.join(script_dir, 'config')
        if not os.path.exists(config_dir_path):
            os.makedirs(config_dir_path)

        # If no file, make it
        args.write_config_to += '.json' if args.write_config_to[-5:] != '.json' else ""
        with open(os.path.join(config_dir_path, args.write_config_to), 'w') as cf:
            json_args = copy.deepcopy(vars(args))
            del json_args['config']
            del json_args['write_config_to']
            json.dump(json_args, cf, indent=4)
            print(f'Successfully wrote the config to config/{args.write_config_to}')

    del args.config
    del args.write_config_to
    return args


def split_qm9(dataset):
    n_train, n_test = 100000, 110000
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:n_test]
    val_dataset = dataset[n_test:]
    return train_dataset, val_dataset, test_dataset


def get_pe(pe_name, pe_dim):
    if 'rw' in pe_name.lower():
        return AddRandomWalkPE(pe_dim)
    elif 'lap' in pe_name.lower():
        # todo: processing doesn't work for lap ?
        return AddLaplacianEigenvectorPE(pe_dim)
    elif 'nope' in pe_name.lower():
        return None
    else:
        raise NotImplementedError(f"PE method \"{pe_name}\" not implemented.")


def get_dataset(dataset_name, pe_name, pe_dim):
    """Gets the corresponding QM9 dataset.
    Dependencies with which data can be loaded:
        torch-cluster==1.6.0, torch-geometric==2.3.0, torch-scatter==1.3.1, torch-sparse==0.6.13, torch==1.13.1"""
    transform = Compose([])
    if 'fc' in dataset_name.lower():
        transform.transforms.append(RadiusGraph(1e6))
    elif 'nope' not in pe_name.lower():
        transform.transforms.append(get_pe(pe_name, pe_dim))
    return QM9(f'./data/{dataset_name}_{args.pe}{args.pe_dim if args.pe != "nope" else ""}',
               pre_transform=transform)


def get_model(model_name):
    if model_name == 'egnn':
        from models.egnn import EGNN
        return EGNN
    elif model_name == 'egnn_lspe':
        from models.egnn_lspe import EGNN
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
    dataset = get_dataset(args.dataset, args.pe, args.pe_dim)

    # Split the dataset into train, val and test
    train_dataset, val_dataset, test_dataset = split_qm9(dataset)

    # Initialize the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Setting the WandB parameters
    config = {
        **vars(args),
        'lspe': 'lspe' in args.model,
        'fc': 'fc' in args.dataset,
    }
    wandb.init(project="dl2-project", entity="msc-ai", config=config, reinit=True,
               name=f'{args.model}_{args.dataset}_{args.pe}{args.pe_dim if args.pe != "nope" else ""}')

    # Initialize the model
    model = get_model(args.model)
    model = model(**vars(args)).to(device)
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
                                 f"_{args.pe}{args.pe_dim if args.pe != 'nope' else ''}" \
                                 f"_epochs-{args.epochs}" \
                                 f"_batch-{args.batch_size}" \
                                 f"_num_hidden-{args.num_hidden}" \
                                 f"_num_layers-{args.num_layers}.pt"
                    # todo if saved_models doesn't exit
                    torch.save(ckpt, model_path)

                # Perform LR step
                scheduler.step()

                # Update the postfix of tqdm with every iteration
                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
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
