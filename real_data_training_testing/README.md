# Real Data Training and Testing System

This module provides a complete pipeline for training Bayesian DeepONet models on PDE datasets. It is independent of the `pde-mol` system.

## Structure

```
real_data_training_testing/
├── solvers/          # PDE solvers (heat1d, moisture1d, burgers1d, custom_pde)
├── configs/          # JSON configuration files for PDEs and training
├── data/             # Generated datasets (.pt files)
├── models/           # Saved trained models
├── test_results/     # Test case visualizations
├── utils/            # Visualization utilities
├── generate_dataset.py    # Dataset generation script
└── train_and_test.py      # Training and testing script
```

## Quick Start

### 1. Generate Dataset

```bash
cd real_data_training_testing
python generate_dataset.py configs/heat1d_config.json
```

This will create `data/heat1d_config_dataset.pt` containing:
- `params`: (num_samples, n_params) - PDE parameters
- `u`: (num_samples, nt, nx) - solutions
- `x`: (nx,) - spatial grid
- `t`: (nt,) - time grid
- `param_names`: list of parameter names

### 2. Train Model

Create a training config file (see `configs/train_config_example.json`):

```json
{
  "dataset_path": "data/heat1d_config_dataset.pt",
  "pde_config_path": "configs/heat1d_config.json",
  "training": {
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "kl_weight": 1e-4,
    "branch_hidden_dims": [128, 128, 128],
    "trunk_hidden_dims": [128, 128, 128],
    "latent_dim": 128,
    "num_test_cases": 4
  }
}
```

Then train:

```bash
python train_and_test.py configs/train_config.json
```

### 3. Results

The script will:
- Train the model with 80/20 train/test split
- Output Test MSE and KL divergence
- Generate test case visualizations:
  - `test_case_XX_comparison.png`: Side-by-side real vs predicted heatmaps
  - `test_case_XX_std.png`: Uncertainty (standard deviation) heatmap

## PDE Solvers

Available solvers in `solvers/`:
- `heat1d.py`: 1D heat equation with periodic BCs
- `moisture1d.py`: 1D moisture diffusion with Robin BCs
- `burgers1d.py`: 1D Burgers equation with periodic BCs
- `custom_pde.py`: Custom advection-diffusion-reaction with Dirichlet BCs
- `robin_general.py`: General Robin boundary conditions

## Configuration Files

### PDE Config Format

```json
{
  "pde_type": "heat1d",
  "equation": "du/dt = nu * d^2u/dx^2",
  "domain": {
    "x": [0.0, 6.28],
    "nx": 128,
    "t": [0.0, 1.0],
    "nt": 400
  },
  "parameters": {
    "nu": {
      "low": 0.1,
      "high": 1.0,
      "description": "Diffusion coefficient"
    }
  },
  "initial_condition": {
    "type": "function",
    "expr": "np.sin(x)"
  },
  "dataset": {
    "num_samples": 100,
    "seed": 42
  }
}
```

### Training Config Format

```json
{
  "dataset_path": "data/dataset.pt",
  "pde_config_path": "configs/pde_config.json",
  "training": {
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "kl_weight": 1e-4,
    "train_ratio": 0.8,
    "T_max": 100,
    "eta_min": 0.0,
    "prior_sigma": 0.1,
    "dropout": 0.1,
    "branch_hidden_dims": [128, 128, 128],
    "trunk_hidden_dims": [128, 128, 128],
    "latent_dim": 128,
    "num_test_cases": 4,
    "test_case_seed": 123
  }
}
```

## Model Architecture

The Bayesian DeepONet consists of:
- **Branch Network**: Takes PDE parameters → latent vector
- **Trunk Network**: Takes (x, t) coordinates → latent vector
- **Output Layer**: Elementwise product of latents → scalar prediction

All layers use Bayesian linear layers with variational inference.

## Features

- **Modular Design**: Each component (solver, model, training) is independent
- **Uncertainty Quantification**: Monte Carlo sampling for prediction uncertainty
- **Visualization**: Automatic generation of comparison and uncertainty plots
- **Configurable**: All hyperparameters in JSON config files

