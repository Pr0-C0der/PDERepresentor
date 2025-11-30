"""
Training and testing script for Bayesian DeepONet on PDE datasets.

Splits data 80/20 train/test, trains with MSE loss + KL divergence,
and evaluates on test set and separate test cases.
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.bayesian_deeponet import BayesianDeepONet
from utils.visualization import plot_heatmap_comparison, plot_std_heatmap
from generate_dataset import solve_pde, load_config


class PDEDataset(Dataset):
    """Dataset for PDE solutions."""
    def __init__(self, dataset_path: str):
        """
        Load dataset from .pt file.

        Expected format:
        - params: (num_samples, n_params) - PDE parameters
        - u: (num_samples, nt, nx) - solutions
        - x: (nx,) - spatial grid
        - t: (nt,) - time grid
        - param_names: list of parameter names
        """
        data = torch.load(dataset_path)
        self.params = data["params"]  # (num_samples, n_params)
        self.u = data["u"]  # (num_samples, nt, nx)
        self.x = data["x"]  # (nx,)
        self.t = data["t"]  # (nt,)
        self.param_names = data["param_names"]
        self.pde_type = data.get("pde_type", "unknown")

        self.num_samples = self.params.shape[0]
        self.nt = self.u.shape[1]
        self.nx = self.u.shape[2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a flattened version for training.

        Returns
        -------
        params : torch.Tensor
            (n_params,) - PDE parameters
        coords : torch.Tensor
            (nt*nx, 2) - spatial-temporal coordinates (x, t)
        values : torch.Tensor
            (nt*nx,) - solution values
        """
        params = self.params[idx]  # (n_params,)
        u_sample = self.u[idx]  # (nt, nx)

        # Create coordinate grid
        # Use indexing="xy" to match solution shape (nt, nx): rows=time, cols=space
        Xg, Tg = np.meshgrid(self.x.numpy(), self.t.numpy(), indexing="xy")  # (nt, nx)
        coords = torch.stack([
            torch.tensor(Xg.flatten(), dtype=torch.float32),
            torch.tensor(Tg.flatten(), dtype=torch.float32),
        ], dim=1)  # (nt*nx, 2)

        # Flatten solution
        values = u_sample.flatten()  # (nt*nx,)

        return params, coords, values


def create_data_loaders(dataset_path: str, train_ratio: float = 0.8, batch_size: int = 32, seed: int = 42):
    """Create train and test data loaders."""
    full_dataset = PDEDataset(dataset_path)
    
    # Split dataset
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, full_dataset


def train_epoch(model, train_loader, optimizer, criterion, kl_weight, device, epoch=None, num_epochs=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    num_batches = 0

    desc = f"Training"
    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch+1}/{num_epochs}"

    for params, coords, values in tqdm(train_loader, desc=desc, leave=False, unit="batch"):
        params = params.to(device)
        coords = coords.to(device)
        values = values.to(device)

        # Expand params to match coords
        batch_size = params.shape[0]
        n_points = coords.shape[1]
        params_expanded = params.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, params.shape[1])
        coords_flat = coords.reshape(-1, coords.shape[2])
        values_flat = values.reshape(-1, 1)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(params_expanded, coords_flat)

        # Loss
        mse_loss = criterion(predictions, values_flat)
        kl = model.kl_divergence()
        loss = mse_loss + kl_weight * kl

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_kl += kl.item()
        num_batches += 1

    return total_loss / num_batches, total_mse / num_batches, total_kl / num_batches


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_mse = 0.0
    total_kl = 0.0
    num_batches = 0

    with torch.no_grad():
        for params, coords, values in tqdm(test_loader, desc="Evaluating", leave=False, unit="batch"):
            params = params.to(device)
            coords = coords.to(device)
            values = values.to(device)

            batch_size = params.shape[0]
            n_points = coords.shape[1]
            params_expanded = params.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, params.shape[1])
            coords_flat = coords.reshape(-1, coords.shape[2])
            values_flat = values.reshape(-1, 1)

            predictions = model(params_expanded, coords_flat)
            mse_loss = criterion(predictions, values_flat)
            kl = model.kl_divergence()

            total_mse += mse_loss.item()
            total_kl += kl.item()
            num_batches += 1

    return total_mse / num_batches, total_kl / num_batches


def train_and_test(config_path: str):
    """
    Main training and testing function.

    Parameters
    ----------
    config_path : str
        Path to training config JSON file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract config values
    dataset_path = config["dataset_path"]
    train_config = config["training"]
    
    # Model hyperparameters
    branch_hidden_dims = train_config["branch_hidden_dims"]
    trunk_hidden_dims = train_config["trunk_hidden_dims"]
    latent_dim = train_config["latent_dim"]
    prior_sigma = train_config.get("prior_sigma", 0.1)
    dropout = train_config.get("dropout", 0.1)

    # Training hyperparameters
    num_epochs = train_config["num_epochs"]
    batch_size = train_config["batch_size"]
    learning_rate = train_config["learning_rate"]
    kl_weight = train_config.get("kl_weight", 1e-4)
    train_ratio = train_config.get("train_ratio", 0.8)
    seed = train_config.get("seed", 42)

    # Scheduler
    T_max = train_config.get("T_max", num_epochs)
    eta_min = train_config.get("eta_min", 0.0)

    # Test cases
    num_test_cases = train_config.get("num_test_cases", 4)
    test_case_seed = train_config.get("test_case_seed", 123)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path)
    n_params = dataset["params"].shape[1]
    print(f"  Number of parameters: {n_params}")
    print(f"  Parameter names: {dataset['param_names']}")

    # Create data loaders
    train_loader, test_loader, full_dataset = create_data_loaders(
        dataset_path, train_ratio=train_ratio, batch_size=batch_size, seed=seed
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Create model
    model = BayesianDeepONet(
        branch_input_dim=n_params,
        trunk_input_dim=2,  # (x, t) or (z, t)
        branch_hidden_dims=branch_hidden_dims,
        trunk_hidden_dims=trunk_hidden_dims,
        latent_dim=latent_dim,
        prior_sigma=prior_sigma,
        dropout=dropout,
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Branch input: {n_params}")
    print(f"  Branch hidden: {branch_hidden_dims}")
    print(f"  Trunk input: 2 (spatial, temporal)")
    print(f"  Trunk hidden: {trunk_hidden_dims}")
    print(f"  Latent dim: {latent_dim}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    best_test_mse = float('inf')
    
    pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    for epoch in pbar:
        train_loss, train_mse, train_kl = train_epoch(
            model, train_loader, optimizer, criterion, kl_weight, device, epoch, num_epochs
        )
        test_mse, test_kl = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_mse < best_test_mse:
            best_test_mse = test_mse

        # Update progress bar with current loss
        pbar.set_postfix({
            'Train Loss': f'{train_loss:.6f}',
            'Test MSE': f'{test_mse:.6f}',
            'LR': f'{scheduler.get_last_lr()[0]:.6f}'
        })

        # Print detailed info after each epoch
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}:")
        tqdm.write(f"  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, KL: {train_kl:.6f})")
        tqdm.write(f"  Test MSE: {test_mse:.6f}, Test KL: {test_kl:.6f}")
        tqdm.write(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\nFinal Test Results:")
    print(f"  Test MSE: {best_test_mse:.6f}")
    final_test_mse, final_test_kl = evaluate(model, test_loader, criterion, device)
    print(f"  Test MSE (final): {final_test_mse:.6f}")
    print(f"  Test KL: {final_test_kl:.6f}")

    # Save model
    model_path = Path(config_path).parent / "models" / f"{Path(dataset_path).stem}_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "test_mse": final_test_mse,
        "test_kl": final_test_kl,
    }, model_path)
    print(f"\nSaved model to {model_path}")

    # Generate test cases
    print(f"\nGenerating {num_test_cases} test cases...")
    pde_config_path = config.get("pde_config_path")
    if pde_config_path is None:
        print("  Warning: pde_config_path not specified, skipping test case generation")
        return

    pde_config = load_config(pde_config_path)
    param_ranges = pde_config["parameters"]
    
    # Sample test parameters
    np.random.seed(test_case_seed)
    test_params_list = []
    for i in range(num_test_cases):
        params = {}
        for name, range_dict in param_ranges.items():
            low = range_dict["low"]
            high = range_dict["high"]
            params[name] = np.random.uniform(low, high)
        test_params_list.append(params)

    # Solve and visualize test cases
    output_dir = Path(config_path).parent / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, test_params in enumerate(tqdm(test_params_list, desc="Generating test cases", unit="case")):
        param_str = ", ".join([f"{k}={v:.4f}" for k, v in test_params.items()])
        tqdm.write(f"\nTest case {i+1}/{num_test_cases}: {param_str}")

        # Solve real PDE
        x, t, u_real = solve_pde(pde_config["pde_type"], test_params, pde_config)
        u_real_tensor = torch.tensor(u_real.T, dtype=torch.float32)  # (nt, nx)

        # Predict with model
        model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32)
        
        # Create coordinate grid
        # x_grid and t_grid should have shape (nt, nx) for proper indexing
        Xg, Tg = np.meshgrid(x, t, indexing="xy")  # (nt, nx)
        coords = torch.stack([
            torch.tensor(Xg.flatten(), dtype=torch.float32),
            torch.tensor(Tg.flatten(), dtype=torch.float32),
        ], dim=1).to(device)  # (nt*nx, 2)

        # Expand params
        params_tensor = torch.tensor([test_params[name] for name in dataset["param_names"]], dtype=torch.float32)
        params_expanded = params_tensor.unsqueeze(0).expand(coords.shape[0], -1).to(device)

        # Predict
        with torch.no_grad():
            u_pred_mean, u_pred_std = model.predict_with_uncertainty(
                params_expanded, coords, num_samples=100
            )

        # Reshape predictions (flattened order matches meshgrid with indexing="xy")
        u_pred_mean = u_pred_mean.cpu().reshape(len(t), len(x)).numpy()
        u_pred_std = u_pred_std.cpu().reshape(len(t), len(x)).numpy()
        
        # u_real is (nx, nt), need to transpose to (nt, nx) for comparison
        if u_real.shape[0] == len(x) and u_real.shape[1] == len(t):
            u_real = u_real.T  # (nx, nt) -> (nt, nx)

        # Plot comparison
        comparison_path = output_dir / f"test_case_{i+1:02d}_comparison.png"
        plot_heatmap_comparison(
            x, t, u_real, u_pred_mean,
            title=f"Test Case {i+1}: {param_str}",
            savepath=comparison_path,
        )

        # Plot std dev
        std_path = output_dir / f"test_case_{i+1:02d}_std.png"
        plot_std_heatmap(
            x, t, u_pred_std,
            title=f"Test Case {i+1} Uncertainty: {param_str}",
            savepath=std_path,
        )

    print(f"\nTest case visualizations saved to {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_and_test.py <train_config.json>")
        sys.exit(1)
    
    train_and_test(sys.argv[1])

