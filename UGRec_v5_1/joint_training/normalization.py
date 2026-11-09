"""Per-item LayerNorm over the AE latent dimensions."""

import math

from torch import nn
from torch.nn import functional as F


class LatentNormalizer(nn.Module):
    def __init__(self, latent_dim, eps=1e-6):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Invalid latent normalization epsilon.")
        self.latent_dim = latent_dim
        self.eps = float(eps)

    def forward(self, latents):
        if latents.shape[-1] != self.latent_dim:
            raise ValueError("Latents have the wrong number of dimensions.")
        return F.layer_norm(latents, (self.latent_dim,), eps=self.eps)
