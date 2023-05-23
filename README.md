## Project Description

## Setting up the Environment
In order to set up the environment for reproducing our experiments, 
install the appropriate conda environment that suits your hardware specifications. 
We put forward two YAML environment files: `environment_gpu.yml` CUDA support and `environment.yml` for CPU (and MPS) support.

```commandline
$ conda env create -f <environment_filename>
```

## Downloading the Data
In all of our experiments, we use the QM9 dataset, first introduced by [Ramakrishnan et al., 2014](https://www.nature.com/articles/sdata201422), comprises approximately 130,000 graphs, each
consisting of around 18 nodes. The objective of analyzing this dataset is to predict 13 quantum chemical properties.
Nevertheless, this study only focuses on inferring the Isotropic Polarizability $\alpha$.

The datasets are automatically downloaded once an experiment 
is run with a specific argument configuration for arguments `--dataset` and `--pe` and `--pe_dim`. 
The arguments can take on the following values:

| Dataset | Explanation                                     |
|---------|-------------------------------------------------|
| QM9     | The original QM9 dataset. **Default**           |  
| QM9_fc  | The fully-connected variant of the QM9 dataset. |

| Positional Encoding | Explanation                                                                                             |
|---------------------|---------------------------------------------------------------------------------------------------------|
| nope                | The dataset is initialized with no PE concatenated to the hidden node state. **Default**                |  
| rw                  | The dataset is initialized with Random-Walk PE concatenated to the hidden node state.                   |
| lap                 | The dataset is initialized with a Laplacian Eigenvector-based PE concatenated to the hidden node state. |

| PE Dimension | Explanation                                              |
|--------------|----------------------------------------------------------|
| [1-28]       | The dimension of the PE vectors per node. **Default 24** |

## Reproducibility of Experiments
We use [WandB](https://wandb.ai/) as our central dashboard to keep track of your hyperparameters, system metrics, and predictions and results.
Before running the experiments, login to your wandb account by entering the following command:
```commandline
$ wandb login 
```

For reproducing the experiments, run the following commands in the terminal after activating your environment.

```commandline
$ python main.py --config mpnn_1.json
```

The training and network parameters for each experiment is stored in a json file in the config/ directory. 
The full path of the config file is not necessary.

Alternatively, instead of the config argument, one can start runs by specifying each individual run argument. For example:

```commandline
python main.py --model mpnn --pe rw --pe_dim 24 --include_dist --lspe
```

One can additionally pass another argument `--write_config_to <new_config_filename>` to write the argument configuration to a file for later convenience when running multiple experiments.
All the running arguments alongside their explanation can be found under `main.py`.

### Output, checkpoints and visualizations

Output results and visualisations are processed directly to WandB. 
TODO: We plan to make available reports with all the run logs via a WandB link.
Checkpoints are stored under `saved_models`.
