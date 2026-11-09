"""Per-item NeXtVLAD tokens: no temporal pooling or concatenation projection.

Group/assignment transforms follow the local NeXtVLAD mechanism, without batch
normalization across history positions. Expansion=groups keeps each cluster at
the original HSTU width. Each item's K output states are summed after HSTU.
"""

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

    def forward(self, x, mask, *, return_assignments=False):
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
        tokens = F.normalize(residual, dim=-1).masked_fill(~mask.bool()[..., None, None], 0.)
        if return_assignments:
            return tokens, assignments, mass
        return tokens


class ItemClusterHistory(nn.Module):
    """Turn each single item vector into K tokens, contextualize, then sum."""

    def __init__(self, dim, clusters, groups=2, expansion=2, assignment_temperature=1.0):
        super().__init__()
        self.tokenizer = ItemNeXtVLAD(dim, clusters, groups, expansion, assignment_temperature)

    def forward(self, x, mask, encoder, *, return_cluster_states=False):
        tokens = self.tokenizer(x, mask)
        batch, items, clusters, dim = tokens.shape
        flat = tokens.reshape(batch, items * clusters, dim)
        flat_mask = mask.repeat_interleave(clusters, dim=1)
        states = encoder(flat, flat_mask, item_group_size=clusters)
        states = states.reshape(batch, items, clusters, dim)
        item_states = states.sum(dim=2)
        return (item_states, states) if return_cluster_states else item_states
