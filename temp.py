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
        # elementwise KL
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
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.layers = nn.ModuleList()
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(BayesianLinear(dims[i], dims[i+1]))
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
    Typical DeepONet: branch_net(u) and trunk_net(y) produce latent vectors of same size,
    output is dot/elementwise-product fed to final BayesianLinear producing scalar output.
    """
    def __init__(self, branch_input_dim, trunk_input_dim,
                 branch_hidden_dims, trunk_hidden_dims, latent_dim,
                 prior_sigma=0.1, dropout=0.1):
        super().__init__()

        # branch and trunk produce latent_dim-sized outputs
        self.branch_net = BayesianMLP(branch_input_dim, branch_hidden_dims, latent_dim, dropout=dropout)
        self.trunk_net = BayesianMLP(trunk_input_dim, trunk_hidden_dims, latent_dim, dropout=dropout)

        # final Bayesian readout (maps latent_dim -> 1)
        self.output_layer = BayesianLinear(latent_dim, 1, prior_sigma=prior_sigma)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, u, y):
        """
        u: branch input (batch, branch_input_dim)
        y: trunk input  (batch, trunk_input_dim)
        returns: (batch, 1)
        """
        branch_output = self.branch_net(u)   # (batch, latent_dim)
        trunk_output = self.trunk_net(y)     # (batch, latent_dim)

        # elementwise product (DeepONet typical)
        combined = branch_output * trunk_output  # broadcast-safe
        output = self.output_layer(combined) + self.b  # (batch, 1)
        return output

    def kl_divergence(self):
        return self.branch_net.kl_divergence() + \
               self.trunk_net.kl_divergence() + \
               self.output_layer.kl_divergence()





def train(self):
    print("Starting DeepONet Training")
    hp_train = self.config['hyperparameters']['training']
    epochs = hp_train['epochs']
    best_loss = float('inf')

    kl_weight_max = hp_train['kl_weight']
    warmup_epochs = hp_train.get('kl_warmup_epochs', 0)
    print(f'kl_weight_max: {kl_weight_max}, warmup_epochs: {warmup_epochs}')

    for epoch in range(epochs):
        self.model.train()
        pbar  = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0

        if warmup_epochs > 0:
            kl_beta = kl_weight_max * min(1.0, (epoch + 1)/warmup_epochs)
        else:
            kl_beta = kl_weight_max

        for conditions, coords, targets in pbar:
            self.optimizer.zero_grad()
            conditions = conditions.to(device)
            coords = coords.to(device)
            targets = targets.to(device)

            batch_size, num_points = coords.shape[0], coords.shape[1]
            branch_in = conditions.unsqueeze(1).repeat(1, num_points, 1).view(-1, 3)
            trunk_in = coords.view(-1, 2)

            y_pred = self.model(branch_in, trunk_in)
            y_true = targets.view(-1 , 1)

            mse_loss = F.mse_loss(y_pred, y_true)
            kl_loss = self.model.kl_divergence()

            loss = mse_loss + kl_beta * kl_loss / len(self.train_dataset)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss/len(self.train_loader)
        self.scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(self.model.state_dict(), "k.pt")
