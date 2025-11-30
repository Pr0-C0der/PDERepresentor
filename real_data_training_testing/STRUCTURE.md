# System Structure

## Overview

This system provides a complete pipeline for training Bayesian DeepONet models on PDE datasets. It is designed to be:
- **Modular**: Each component is independent and testable
- **Simple**: Clear, straightforward implementations
- **Extensible**: Easy to add new PDE types
- **Safe**: No raw eval in critical paths (except for initial conditions which are controlled)

## Directory Structure

```
real_data_training_testing/
├── solvers/              # PDE solvers (standalone functions)
│   ├── heat1d.py         # 1D heat equation (periodic BCs)
│   ├── moisture1d.py     # 1D moisture diffusion (Robin BCs)
│   ├── burgers1d.py      # 1D Burgers equation (periodic BCs)
│   ├── custom_pde.py     # Custom advection-diffusion-reaction (Dirichlet BCs)
│   └── robin_general.py  # General Robin BC solver
│
├── configs/              # JSON configuration files
│   ├── heat1d_config.json
│   ├── moisture1d_config.json
│   ├── burgers1d_config.json
│   ├── custom_pde_config.json
│   └── train_config_example.json
│
├── models/               # Neural network models
│   └── bayesian_deeponet.py  # Generalized Bayesian DeepONet
│
├── utils/                # Utility modules
│   └── visualization.py  # Plotting functions
│
├── data/                 # Generated datasets (.pt files)
├── models/               # Saved trained models
├── test_results/         # Test case visualizations
│
├── generate_dataset.py   # Dataset generation script
├── train_and_test.py     # Training and testing script
├── requirements.txt      # Python dependencies
└── README.md            # User documentation
```

## Data Flow

1. **Configuration** → JSON files define PDE parameters and training hyperparameters
2. **Dataset Generation** → `generate_dataset.py` samples parameters, solves PDEs, saves .pt files
3. **Training** → `train_and_test.py` loads dataset, trains model, evaluates on test set
4. **Visualization** → Test cases generate comparison and uncertainty plots

## Key Components

### PDE Solvers (`solvers/`)

Each solver is a standalone module with a `solve_*` function:
- Takes PDE parameters and domain configuration
- Returns spatial grid, time grid, and solution array
- Uses appropriate numerical methods (Crank-Nicolson, implicit Euler, explicit Euler)

### Dataset Generation (`generate_dataset.py`)

- Loads PDE config JSON
- Samples parameters from specified ranges
- Solves PDE for each sample
- Saves as PyTorch tensor file with:
  - `params`: (num_samples, n_params)
  - `u`: (num_samples, nt, nx)
  - `x`: (nx,)
  - `t`: (nt,)
  - `param_names`: list

### Model (`models/bayesian_deeponet.py`)

- **BayesianLinear**: Variational Bayesian linear layer
- **BayesianMLP**: Multi-layer perceptron with Bayesian layers
- **BayesianDeepONet**: Operator learning network
  - Branch: PDE parameters → latent vector
  - Trunk: (x, t) coordinates → latent vector
  - Output: elementwise product → scalar

### Training (`train_and_test.py`)

- 80/20 train/test split
- MSE loss + KL divergence regularization
- AdamW optimizer
- CosineAnnealingLR scheduler
- Generates test case visualizations

### Visualization (`utils/visualization.py`)

- `plot_heatmap_comparison`: Side-by-side real vs predicted
- `plot_std_heatmap`: Uncertainty visualization

## Configuration Schema

### PDE Config

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

### Training Config

```json
{
  "dataset_path": "data/dataset.pt",
  "pde_config_path": "configs/pde_config.json",
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

## Usage Example

```bash
# 1. Generate dataset
python generate_dataset.py configs/heat1d_config.json

# 2. Train model
python train_and_test.py configs/train_config.json

# 3. Check results
# - Model saved in models/
# - Test visualizations in test_results/
```

## Extensibility

To add a new PDE type:

1. Create solver in `solvers/new_pde.py` with `solve_new_pde()` function
2. Add case in `generate_dataset.py` `solve_pde()` function
3. Create config JSON in `configs/`
4. Generate dataset and train as usual

The system is designed to handle any 1D PDE with appropriate boundary conditions.

