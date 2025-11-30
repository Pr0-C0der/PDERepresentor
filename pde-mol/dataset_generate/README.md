# Dataset Generation Examples

This folder contains JSON templates for parameterized PDE problems that can be used for dataset generation.

## Usage

### Using the CLI

```bash
# Generate dataset from a template
python run_problem.py dataset_generate/heat1d_parameterized.json

# Override number of samples
python run_problem.py dataset_generate/heat1d_parameterized.json --samples 100

# Override save path
python run_problem.py dataset_generate/heat1d_parameterized.json --save my_dataset.pt

# Force dataset mode even if JSON doesn't have dataset.enabled=true
python run_problem.py dataset_generate/heat1d_parameterized.json --dataset
```

### Using Python API

```python
from pde.dataset import ParameterRange, generate_dataset

# Define parameter ranges
param_ranges = [
    ParameterRange("H", 0.1, 1.0),
    ParameterRange("A0", 0.5, 2.0),
]

# Generate dataset
dataset = generate_dataset(
    json_template="dataset_generate/heat1d_parameterized.json",
    param_ranges=param_ranges,
    num_samples=50,
    savepath="data/my_dataset.pt",
    seed=42,
)

# Load dataset in PyTorch
import torch
data = torch.load("data/my_dataset.pt")
print(f"Parameters shape: {data['params'].shape}")
print(f"Solution shape: {data['u'].shape}")
```

## Template Format

Templates use parameter placeholders (e.g., `"H"`, `"A0"`) instead of numeric values. These are replaced with sampled values during dataset generation.

### Example Template

```json
{
  "domain": {
    "type": "1d",
    "x0": 0.0,
    "x1": 6.283185307179586,
    "nx": 101,
    "periodic": true
  },
  "initial_condition": {
    "type": "expression",
    "expr": "A0 * np.sin(x)"
  },
  "operators": [
    {
      "type": "diffusion",
      "nu": "H"
    }
  ],
  "time": {
    "t0": 0.0,
    "t1": 1.0,
    "num_points": 50
  },
  "dataset": {
    "enabled": true,
    "num_samples": 50,
    "parameters": {
      "H": [0.1, 1.0],
      "A0": [0.5, 2.0]
    },
    "savepath": "data/dataset.pt",
    "seed": 42,
    "save_heatmaps": true,
    "overwrite": false
  }
}
```

## Available Templates

### 1D Problems

- `heat1d_parameterized.json` - 1D heat equation with parameterized diffusion coefficient (H) and initial amplitude (A0)
- `burgers1d_parameterized.json` - 1D Burgers equation with parameterized viscosity (nu) and initial amplitude (A0)
- `moisture1d_parameterized.json` - 1D moisture diffusion with Robin BCs, parameterized diffusion (D), heat transfer (h), environmental value (u_env), and initial condition (X0)
- `moisture1d_dirichlet_parameterized.json` - 1D moisture diffusion with Dirichlet BCs, parameterized diffusion (D), boundary value (u_boundary), and initial condition (X0)
- `custom1d_pde_parameterized.json` - Custom 1D PDE with advection-diffusion-reaction, parameterized advection (a), diffusion (nu), reaction (lam), and initial condition width (sigma)
- `robin_general_example_parameterized.json` - General Robin BC example with parameterized diffusion (D), initial amplitude (A0), and Robin BC coefficients (a_left, b_left, c_left, a_right, b_right, c_right)

### 2D Problems

- `heat2d_parameterized.json` - 2D heat equation with parameterized diffusion coefficient (H) and initial amplitude (A0)
- `burgers2d_parameterized.json` - 2D Burgers equation with parameterized viscosity (nu) and initial amplitude (A0)
- `reaction_diffusion2d_parameterized.json` - 2D reaction-diffusion equation with parameterized diffusion (D), reaction rate (r), initial condition center (x0, y0), and width (sigma)
- `custom2d_pde_parameterized.json` - Custom 2D PDE with advection-diffusion-reaction, parameterized diffusion (nu), advection (a_x, a_y), reaction (lam), initial condition center (x0, y0), and width (sigma)

## Dataset Format

The generated `.pt` file contains:

- `params`: Tensor of shape `(num_samples, num_params)` - parameter values
- `param_names`: List of parameter names in order
- `x`: Tensor of shape `(nx,)` for 1D or `(ny, nx)` for 2D - spatial grid
- `t`: Tensor of shape `(nt,)` - time points
- `u`: Tensor of shape `(num_samples, nt, nx)` for 1D or `(num_samples, nt, ny, nx)` for 2D - solutions

## Heatmap Visualization

When `save_heatmaps: true` is set in the dataset configuration (default), heatmaps are automatically saved for each sample:

- **1D problems**: `u(x,t)` heatmaps saved as `sample_XXXX_heatmap.png` in `dataset_generate_plots/{problem_name}/`
- **2D problems**: Final state heatmaps saved as `sample_XXXX_final.png` in `dataset_generate_plots/{problem_name}/`

Each heatmap filename includes the sample number and the plot title includes the parameter values used for that sample. This allows easy verification of the generated dataset.

## Dataset Caching

By default, if a dataset file already exists at the specified `savepath`, the generation is skipped and the existing dataset is loaded instead. This prevents unnecessary regeneration of datasets.

To force regeneration, set `"overwrite": true` in the dataset configuration, or delete the existing `.pt` file.

