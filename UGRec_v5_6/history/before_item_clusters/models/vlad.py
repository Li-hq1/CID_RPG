"""Padding-aware NeXtVLAD descriptors for every causal history prefix.

Adapted from the K8/S1 exploration in representation_structure_20261105.
Assignments and group gates are learned; centers are not fitted beforehand.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CausalNeXtVLAD(nn.Module):
    def __init__(self, dim, clusters, groups=8, expansion=2, assignment_temperature=1.0):
        super().__init__()
        if min(dim, clusters, groups, expansion) <= 0 or expansion * dim % groups:
            raise ValueError("VLAD dimensions must be positive and expansion*dim divisible by groups.")
        self.K, self.G, self.S = clusters, groups, expansion * dim // groups
        if assignment_temperature <= 0:
            raise ValueError("VLAD assignment temperature must be positive.")
        self.assignment_temperature = assignment_temperature
        self.fc0 = nn.Linear(dim, expansion * dim)
        self.fc_gk = nn.Linear(expansion * dim, groups * clusters)
        self.fc_g = nn.Linear(expansion * dim, groups)
        self.centers = nn.Parameter(torch.randn(clusters, self.S) * 0.02)

    def forward(self, x, mask):
        expanded = self.fc0(x)
        grouped = expanded.reshape(*x.shape[:2], self.G, self.S)
        scores = self.fc_gk(expanded).reshape(*x.shape[:2], self.G, self.K)
        assign = (scores / self.assignment_temperature).softmax(-1)
        gate = self.fc_g(expanded).sigmoid() * mask[..., None].to(x.dtype)
        activation = assign * gate[..., None]
        weighted = torch.einsum("btgk,btgs->btks", activation, grouped).cumsum(1)
        mass = activation.sum(2).cumsum(1)
        residual = weighted - mass[..., None] * self.centers
        return F.normalize(residual, dim=-1), mass
