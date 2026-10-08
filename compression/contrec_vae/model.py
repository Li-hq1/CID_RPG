"""SigmaVae adapted from ContRec/code/rec_tokenizer/vae.py (not a KL-regularized VAE)."""

import torch
from torch import nn
from torch.nn import functional as F


def configure_loss(x, x_hat, mus, encoded_embeds, *, beta=0.5, gamma=0.5, twins_lambda=0.005):
    """ContRec reconstruction + μ-norm + Barlow twins (anti-collapse) loss.

    Matches ContRec/code/rec_tokenizer/vae.py::configure_loss; std is clamped for stability.
    Requires at least two encoder branches for the twins term.
    """
    if len(encoded_embeds) < 2:
        raise ValueError("Barlow twins loss requires at least two encoder branches.")
    batch, dim = encoded_embeds[0].shape
    if batch < 2:
        raise ValueError("Barlow twins loss requires batch size >= 2.")

    reconstruction_loss = F.mse_loss(x_hat, x)

    # Larger latent variance: penalize the average encoder mean (ContRec).
    avg_mu = torch.stack(mus, dim=0).mean(dim=0)
    mu_loss = torch.linalg.vector_norm(avg_mu) / batch

    # Barlow Twins between the first two noisy branches.
    z_norm = []
    for embed in encoded_embeds:
        std = embed.std(0).clamp_min(1e-6)
        z_norm.append((embed - embed.mean(0)) / std)
    cross = z_norm[0].T @ z_norm[1] / dim
    target = torch.eye(dim, device=x.device, dtype=x.dtype)
    c_diff = (cross - target).pow(2)
    off_diagonal = c_diff * twins_lambda
    off_diagonal.diagonal().copy_(c_diff.diagonal())
    twins_loss = off_diagonal.sum() / dim

    total = reconstruction_loss + beta * mu_loss + gamma * twins_loss
    return {
        "loss": total,
        "reconstruction": reconstruction_loss.detach(),
        "mu": mu_loss.detach(),
        "twins": twins_loss.detach(),
    }


class SigmaVae(nn.Module):
    def __init__(self, input_dim, dim=2048, enc_num=2, embeds_num=1,
                 mask_ratio=0.1, sigma=0.1, seed=42):
        super().__init__()
        if min(input_dim, dim, enc_num, embeds_num) <= 0:
            raise ValueError("Model dimensions and embeds_num must be positive.")
        if not 0 <= mask_ratio <= 1 or sigma < 0:
            raise ValueError("Invalid mask_ratio or sigma.")
        self.dim = dim
        self.enc_num = enc_num
        self.mask_ratio = mask_ratio
        # Preserve ContRec's one-time noise draw without resetting the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.register_buffer("epsilon", torch.randn(1, dim))
            self.register_buffer("sigma", torch.normal(0, sigma, size=(1,)))
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
                    nn.Dropout(0.5), nn.ReLU(),
                    nn.Linear(128, 256), nn.BatchNorm1d(256),
                    nn.Dropout(0.2), nn.ReLU(), nn.Linear(256, dim),
                ) for _ in range(enc_num)
            ])
            self.decoder = nn.Sequential(
                nn.Linear(dim, 256), nn.BatchNorm1d(256),
                nn.Dropout(0.5), nn.ReLU(),
                nn.Linear(256, 128), nn.BatchNorm1d(128),
                nn.Dropout(0.2), nn.ReLU(), nn.Linear(128, input_dim),
            )
            self.pos = nn.Embedding(1, input_dim)
            self.pos.weight.data.uniform_(-1.0 / embeds_num, 1.0 / embeds_num)

    def encode(self, x):
        """Return both branches; inference callers must put the model in eval mode."""
        mus = [encoder(x) for encoder in self.encoders]
        return mus, [mu + self.sigma * self.epsilon for mu in mus]

    def latent(self, x):
        return torch.stack(self.encode(x)[1]).mean(dim=0)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        if self.training:
            # Shared feature mask, as in ContRec, without mutating a reconstruction target.
            mask = torch.rand(x.shape[1], device=x.device) < self.mask_ratio
            x = x.masked_fill(mask, 0) + self.pos.weight
        mus, encoded = self.encode(x)
        return self.decode(torch.stack(encoded).mean(dim=0)), mus, encoded

    @torch.no_grad()
    def valid(self, x):
        if self.training:
            raise RuntimeError("Call eval() before validation to disable Dropout and masking.")
        return self(x)
