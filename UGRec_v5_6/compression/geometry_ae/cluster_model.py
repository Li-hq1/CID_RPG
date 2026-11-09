"""Multi-vector AE with an explicit shared decoder/catalog item representation."""

import torch
from torch import nn
from torch.nn import functional as F

from models.item_vlad import ItemNeXtVLAD
from .model import GeometryAE


class ClusterAggregation(nn.Module):
    """A sum, or one linear concat map initialized with identity blocks."""

    def __init__(self, dim, clusters, mode="sum"):
        super().__init__()
        if mode not in {"sum", "linear"}:
            raise ValueError("AE cluster aggregation must be sum or linear.")
        self.mode, self.dim, self.clusters = mode, dim, clusters
        if mode == "linear":
            self.weight = nn.Parameter(torch.eye(dim).repeat(1, clusters))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, components):
        if components.shape[-2:] != (self.clusters, self.dim):
            raise ValueError("Cluster aggregation needs [...,K,D] components.")
        if self.mode == "sum":
            return components.sum(-2)
        return F.linear(components.flatten(-2), self.weight, self.bias)


class ClusterGeometryAE(GeometryAE):
    def __init__(self, input_dim, config, input_mean=None, input_scale=None):
        super().__init__(input_dim, config, input_mean, input_scale)
        # Preserve the baseline backbone/decoder and the caller's recommender RNG.
        # Sum and learned aggregation have identical common initial parameters.
        with torch.random.fork_rng(devices=[]):
            self.cluster_tokenizer = ItemNeXtVLAD(config.latent_dim, config.cluster_count,
                config.cluster_groups, config.cluster_expansion, config.cluster_assignment_temperature)
            self.cluster_aggregation = ClusterAggregation(config.latent_dim, config.cluster_count,
                                                         config.cluster_aggregation)

    def components(self, x, *, return_details=False):
        backbone = super().latent(x)
        mask = torch.ones(len(x), 1, device=x.device, dtype=torch.bool)
        result = self.cluster_tokenizer(backbone[:, None], mask,
                                        return_assignments=return_details, normalize=False)
        if return_details:
            residual, assignment, mass = result
            return residual[:,0], assignment[:,0], mass[:,0]
        return result[:,0]

    def latent(self, x):
        # Both GeometryAE.forward/decode and JointModel.encode_ids use this z.
        return self.cluster_aggregation(self.components(x))
