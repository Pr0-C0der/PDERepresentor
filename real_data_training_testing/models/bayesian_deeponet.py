"""
Generalized Bayesian DeepONet for PDE operator learning.

Adapted from temp.py to work with any PDE data structure.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """
    Bayesian fully-connected layer with mean/rho parameterization.
    weight ~ N(weight_mu, (softplus(weight_rho)+eps)^2)
    bias   ~ N(bias_mu,   (softplus(bias_rho)+eps)^2)
    """
    def __init__(self, in_features, out_features, prior_sigma=0.1, eps=1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = float(prior_sigma)
        self.prior_log_sigma = math.log(self.prior_sigma)
        self.eps = eps

        # variational parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(-3.0)   # softplus(-3) ~ small sigma ~ 0.05
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_rho.data.fill_(-3.0)

    def forward(self, x):
        """
        x: (batch, in_features) or (..., in_features)
        returns: (batch, out_features)
        """
        if self.training:
            weight_sigma = F.softplus(self.weight_rho) + self.eps
            bias_sigma = F.softplus(self.bias_rho) + self.eps
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + weight_sigma * weight_eps
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """
        KL[q(w) || p(w)] where p = Normal(0, prior_sigma^2)
        KL for each scalar: ln(sigma_p / sigma_q) + (sigma_q^2 + mu_q^2)/(2 sigma_p^2) - 1/2
        Sum over all parameters.
        """
        device = self.weight_mu.device

        weight_sigma = F.softplus(self.weight_rho) + self.eps
        bias_sigma = F.softplus(self.bias_rho) + self.eps

        # convert prior log-sigma to tensor on correct device
        prior_log_sigma = torch.tensor(self.prior_log_sigma, device=device, dtype=self.weight_mu.dtype)
        prior_sigma_sq = float(self.prior_sigma ** 2)

        # weight KL
        weight_log_sigma = torch.log(weight_sigma)
        kl_weight = torch.sum(
            (prior_log_sigma - weight_log_sigma) +
            (weight_sigma.pow(2) + self.weight_mu.pow(2)) / (2.0 * prior_sigma_sq) -
            0.5
        )

        # bias KL
        bias_log_sigma = torch.log(bias_sigma)
        kl_bias = torch.sum(
            (prior_log_sigma - bias_log_sigma) +
            (bias_sigma.pow(2) + self.bias_mu.pow(2)) / (2.0 * prior_sigma_sq) -
            0.5
        )

        return kl_weight + kl_bias


class BayesianMLP(nn.Module):
    """
    Simple MLP composed of BayesianLinear layers separated by LayerNorm + GELU + Dropout
    hidden_dims: list of ints (hidden layer sizes)
    output_dim: final output dimension (e.g. latent dim)
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1, prior_sigma=0.1):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.layers = nn.ModuleList()
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(BayesianLinear(dims[i], dims[i+1], prior_sigma=prior_sigma))
            if i < len(dims) - 2:
                # add normalization and activation as separate modules to keep ModuleList simple
                self.layers.append(nn.LayerNorm(dims[i+1]))
                self.layers.append(nn.GELU())
                self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        # sequentially apply modules in ModuleList
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def kl_divergence(self):
        kl = 0.0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_divergence()
        return kl


class BayesianDeepONet(nn.Module):
    """
    Generalized Bayesian DeepONet for PDE operator learning.

    Branch network: takes input function (parameterized by PDE parameters or function values)
    Trunk network: takes spatial-temporal coordinates (x, t) or (z, t)
    Output: scalar value u(x, t) or X(z, t)

    This version is generalized to work with:
    - Any number of PDE parameters (branch_input_dim = n_params)
    - Any spatial dimension (trunk_input_dim = 2 for 1D spatial + time)
    """
    def __init__(self, branch_input_dim, trunk_input_dim,
                 branch_hidden_dims, trunk_hidden_dims, latent_dim,
                 prior_sigma=0.1, dropout=0.1):
        super().__init__()

        # branch and trunk produce latent_dim-sized outputs
        self.branch_net = BayesianMLP(
            branch_input_dim, branch_hidden_dims, latent_dim,
            dropout=dropout, prior_sigma=prior_sigma
        )
        self.trunk_net = BayesianMLP(
            trunk_input_dim, trunk_hidden_dims, latent_dim,
            dropout=dropout, prior_sigma=prior_sigma
        )

        # final Bayesian readout (maps latent_dim -> 1)
        self.output_layer = BayesianLinear(latent_dim, 1, prior_sigma=prior_sigma)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, u, y):
        """
        Forward pass.

        Parameters
        ----------
        u : torch.Tensor
            Branch input (batch, branch_input_dim) - PDE parameters
        y : torch.Tensor
            Trunk input (batch, trunk_input_dim) - spatial-temporal coordinates (x, t) or (z, t)

        Returns
        -------
        output : torch.Tensor
            (batch, 1) - predicted solution values
        """
        branch_output = self.branch_net(u)   # (batch, latent_dim)
        trunk_output = self.trunk_net(y)     # (batch, latent_dim)

        # elementwise product (DeepONet typical)
        combined = branch_output * trunk_output  # broadcast-safe
        output = self.output_layer(combined) + self.b  # (batch, 1)
        return output

    def kl_divergence(self):
        """Compute total KL divergence for all Bayesian layers."""
        return self.branch_net.kl_divergence() + \
               self.trunk_net.kl_divergence() + \
               self.output_layer.kl_divergence()

    def predict_with_uncertainty(self, u, y, num_samples=100):
        """
        Predict with uncertainty estimation using Monte Carlo sampling.

        Parameters
        ----------
        u : torch.Tensor
            Branch input (batch, branch_input_dim)
        y : torch.Tensor
            Trunk input (batch, trunk_input_dim)
        num_samples : int
            Number of Monte Carlo samples

        Returns
        -------
        mean : torch.Tensor
            Mean predictions (batch, 1)
        std : torch.Tensor
            Standard deviation of predictions (batch, 1)
        """
        self.train()  # Enable dropout and sampling
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(u, y)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch, 1)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std

