"""Per-item NeXtVLAD tokens: no temporal pooling or concatenation projection.

Group/assignment transforms follow the local NeXtVLAD mechanism, without batch
normalization across history positions. Expansion=groups keeps each cluster at
the original HSTU width. Per-item contextual pooling defaults to the legacy sum.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ItemNeXtVLAD(nn.Module):
    def __init__(self, dim, clusters, groups=2, expansion=2, assignment_temperature=1.0):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (dim, clusters, groups, expansion)):
            raise ValueError("VLAD dimensions must be positive integers.")
        if expansion != groups:
            raise ValueError("Expansion must equal groups to preserve width without output projection.")
        if assignment_temperature <= 0:
            raise ValueError("Assignment temperature must be positive.")
        self.dim, self.K, self.G = dim, clusters, groups
        self.assignment_temperature = assignment_temperature
        self.fc0 = nn.Linear(dim, expansion * dim)
        self.fc_gk = nn.Linear(expansion * dim, groups * clusters)
        self.fc_g = nn.Linear(expansion * dim, groups)
        self.centers = nn.Parameter(torch.randn(clusters, dim) * 0.02)

    def forward(self, x, mask, *, return_assignments=False, normalize=True):
        if x.ndim != 3 or x.shape[-1] != self.dim or mask.shape != x.shape[:2]:
            raise ValueError("Expected [B,T,D] item vectors and [B,T] validity mask.")
        expanded = self.fc0(x)
        grouped = expanded.reshape(*x.shape[:2], self.G, self.dim)
        scores = self.fc_gk(expanded).reshape(*x.shape[:2], self.G, self.K)
        assignments = (scores / self.assignment_temperature).softmax(-1)
        activation = assignments * self.fc_g(expanded).sigmoid()[..., None]
        # No cumsum: every descriptor depends on this item only.
        mass = activation.sum(-2)
        weighted = torch.einsum("btgk,btgd->btkd", activation, grouped)
        residual = weighted - mass[..., None] * self.centers
        tokens = (F.normalize(residual, dim=-1) if normalize else residual)
        tokens = tokens.masked_fill(~mask.bool()[..., None, None], 0.)
        if return_assignments:
            return tokens, assignments, mass
        return tokens


class ItemResidualTokens(nn.Module):
    """Single-descriptor NetVLAD directions with canceled assignments removed.

    normalize(a_k(x) * (x-c_k)) == normalize(x-c_k) for positive a_k.
    This is a directional center bank, not learned soft routing of descriptors.
    """

    def __init__(self, dim, clusters, centered=True):
        super().__init__()
        if type(dim) is not int or type(clusters) is not int or min(dim, clusters) <= 0:
            raise ValueError("Token dimensions must be positive integers.")
        if not centered and clusters != 1:
            raise ValueError("Normalized input control requires exactly one token.")
        self.dim, self.K, self.G = dim, clusters, 1
        self.centers = nn.Parameter(torch.randn(clusters, dim) * .02) if centered else None

    def forward(self, x, mask, *, return_assignments=False, normalize=True):
        if x.ndim != 3 or x.shape[-1] != self.dim or mask.shape != x.shape[:2]:
            raise ValueError("Expected [B,T,D] item vectors and [B,T] validity mask.")
        residual = x[..., None, :]
        if self.centers is not None:
            residual = residual - self.centers
        tokens = F.normalize(residual, dim=-1) if normalize else residual
        tokens = tokens.masked_fill(~mask.bool()[..., None, None], 0.)
        if return_assignments:
            assignments = x.new_full((*x.shape[:2], 1, self.K), 1. / self.K)
            return tokens, assignments, assignments.squeeze(-2)
        return tokens


class SharedTokenMapping(nn.Module):
    """Bias-free per-token mapping, identity initialized without consuming RNG."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim))

    def forward(self, tokens):
        return F.linear(tokens, self.weight)


class ItemClusterAggregation(nn.Module):
    """Symmetric item-local pooling, with K-scaled weights to include sum."""

    def __init__(self, dim, mode="sum"):
        super().__init__()
        if mode not in {"sum", "mass", "context"}:
            raise ValueError("Item cluster pooling must be sum, mass or context.")
        self.mode = mode
        if mode == "context":
            # No RNG draw: preserve the tokenizer and shared backbone initial state.
            self.query = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("query", None)

    def forward(self, states, mass=None, mask=None, *, return_weights=False):
        k = states.shape[-2]
        if self.mode == "sum":
            weights = states.new_full(states.shape[:-1], 1. / k)
            pooled = states.sum(-2)
        else:
            if self.mode == "mass":
                if mass is None or mass.shape != states.shape[:-1]:
                    raise ValueError("Mass pooling requires matching per-cluster activation mass.")
                weights = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
            else:
                normalized = F.layer_norm(states, (states.shape[-1],), eps=1e-6)
                scores = (normalized * self.query).sum(-1)
                weights = scores.softmax(-1)
            pooled = (states * (k * weights)[..., None]).sum(-2)
        if mask is not None:
            pooled = pooled.masked_fill(~mask.bool()[..., None], 0.)
            weights = weights.masked_fill(~mask.bool()[..., None], 0.)
        return (pooled, weights) if return_weights else pooled


class ItemClusterHistory(nn.Module):
    """Turn one item vector into K tokens, contextualize, then pool per item."""

    def __init__(self, dim, clusters, groups=2, expansion=2, assignment_temperature=1.0,
                 scale="unit", pooling="sum", scale_multiplier=1.,
                 variant="nextvlad", mapping="none"):
        super().__init__()
        if scale not in {"unit", "sqrt_dim"}:
            raise ValueError("Cluster token scale must be unit or sqrt_dim.")
        if not math.isfinite(scale_multiplier) or scale_multiplier <= 0:
            raise ValueError("Cluster token scale multiplier must be finite and positive.")
        self.token_scale = (math.sqrt(dim) if scale == "sqrt_dim" else 1.) * scale_multiplier
        if variant not in {"nextvlad", "residual", "normalized_input"}:
            raise ValueError("Unknown item token variant.")
        if mapping not in {"none", "linear"}:
            raise ValueError("Unknown item token mapping.")
        if variant != "nextvlad" and (groups != 1 or expansion != 1 or pooling != "sum"):
            raise ValueError("Ungrouped tokens require groups=expansion=1 and sum pooling.")
        self.variant, self.mapping = variant, mapping
        self.tokenizer = (ItemNeXtVLAD(dim, clusters, groups, expansion, assignment_temperature)
                          if variant == "nextvlad" else
                          ItemResidualTokens(dim, clusters, centered=variant == "residual"))
        self.token_mapping = SharedTokenMapping(dim) if mapping == "linear" else nn.Identity()
        self.pool = ItemClusterAggregation(dim, pooling)

    def prepare_tokens(self, x, mask, *, return_details=False):
        tokens, assignments, mass = self.tokenizer(
            x, mask, return_assignments=True, normalize=self.mapping == "none")
        if self.mapping != "none":
            tokens = F.normalize(self.token_mapping(tokens), dim=-1)
        tokens = (tokens * self.token_scale).masked_fill(~mask.bool()[..., None, None], 0.)
        return (tokens, assignments, mass) if return_details else tokens

    def forward(self, x, mask, encoder, *, return_cluster_states=False):
        tokens, _, mass = self.prepare_tokens(x, mask, return_details=True)
        batch, items, clusters, dim = tokens.shape
        flat = tokens.reshape(batch, items * clusters, dim)
        flat_mask = mask.repeat_interleave(clusters, dim=1)
        states = encoder(flat, flat_mask, item_group_size=clusters)
        states = states.reshape(batch, items, clusters, dim)
        item_states = self.pool(states, mass, mask)
        return (item_states, states) if return_cluster_states else item_states
