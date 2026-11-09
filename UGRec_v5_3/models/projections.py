"""Explicit transition ablations; each default preserves the v5 state keys."""

from torch import nn

from models.tigerdiff import InputProj


def input_projection(left, right, mode):
    if mode == "linear_norm":
        return InputProj(left, right)
    if mode == "linear":
        return nn.Linear(left, right)
    if left != right:
        raise ValueError("An input projection without a linear needs equal dimensions.")
    return nn.LayerNorm(right, eps=1e-8) if mode == "norm" else nn.Identity()


def condition_projection(width, mode, depth=1, norm="none"):
    if mode == "norm_linear":
        return nn.Sequential(nn.LayerNorm(width, eps=1e-8), nn.Linear(width, width))
    if mode == "linear":
        return nn.Linear(width, width)
    if mode == "norm":
        return nn.LayerNorm(width, eps=1e-8)
    if mode == "identity":
        return nn.Identity()
    layers = []
    for index in range(depth):
        layers.append(nn.Linear(width, width))
        if norm == "layer":
            layers.append(nn.LayerNorm(width, eps=1e-8))
        if index < depth - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)
