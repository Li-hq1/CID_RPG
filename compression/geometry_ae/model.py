"""Deterministic autoencoders and item-only geometry objectives."""

import torch
from torch import nn
from torch.nn import functional as F


class GeometryAE(nn.Module):
    def __init__(self, input_dim, config, input_mean=None, input_scale=None):
        super().__init__()
        config.validate()
        self.register_buffer("input_mean", torch.zeros(input_dim) if input_mean is None else input_mean.clone())
        self.register_buffer("input_scale", torch.tensor(1.) if input_scale is None else input_scale.clone())
        widths = [input_dim, *([] if config.architecture == "linear" else config.hidden_dims), config.latent_dim]

        def network(dims):
            layers = []
            for index, (left, right) in enumerate(zip(dims, dims[1:])):
                layers.append(nn.Linear(left, right))
                if index < len(dims) - 2:
                    layers.append(nn.GELU())
            return nn.Sequential(*layers)

        self.encoder = network(widths)
        self.decoder = network(widths[::-1])

    def latent(self, x):
        return self.encoder((x - self.input_mean) / self.input_scale)

    def decode(self, z):
        return self.decoder(z) * self.input_scale + self.input_mean

    def forward(self, x):
        z = self.latent(x)
        return self.decode(z), z


def standardize_batch(z):
    centered = z - z.mean(0)
    std = centered.square().mean(0).add(1e-8).sqrt()
    return centered / std, std


def relation_kl(x, z, temperature=0.1):
    if len(x) < 2:
        return z.sum() * 0.0
    teacher = F.normalize(x.detach(), dim=1)
    student = F.normalize(standardize_batch(z)[0], dim=1)
    diagonal = torch.eye(len(x), device=x.device, dtype=torch.bool)
    # A finite sentinel avoids 0 * -inf in KL while excluding self matches.
    teacher_logits = ((teacher @ teacher.T) / temperature).masked_fill(diagonal, -1e4)
    student_logits = ((student @ student.T) / temperature).masked_fill(diagonal, -1e4)
    return F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1),
                    reduction="batchmean")


def local_relation_kl(x, z, candidate_x, candidate_z, mean, std, temperature=0.1):
    teacher = F.normalize(x.detach(), dim=-1)
    teacher_candidates = F.normalize(candidate_x.detach(), dim=-1)
    student = F.normalize((z - mean.detach()) / std.detach().clamp_min(1e-6), dim=-1)
    student_candidates = F.normalize(
        (candidate_z - mean.detach()) / std.detach().clamp_min(1e-6), dim=-1)
    teacher_logits = (teacher[:, None, :] * teacher_candidates).sum(-1) / temperature
    student_logits = (student[:, None, :] * student_candidates).sum(-1) / temperature
    return F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1),
                    reduction="batchmean")


def loss_terms(x, reconstructed, z, scale, temperature=0.1, *, batch_relation=True):
    normalized, std = standardize_batch(z)
    correlation = normalized.T @ normalized / len(z)
    off_diagonal = correlation[~torch.eye(z.shape[1], device=z.device, dtype=torch.bool)]
    return {
        "reconstruction": F.mse_loss(reconstructed, x) / scale.square(),
        "cosine": (1 - F.cosine_similarity(reconstructed, x, dim=1)).mean(),
        "variance": F.relu(1 - std).square().mean(),
        "correlation": off_diagonal.square().mean() if off_diagonal.numel() else z.sum() * 0,
        "relation": relation_kl(x, z, temperature) if batch_relation else z.sum() * 0.,
    }


def objective_weights(config, epoch):
    ramp = min(1., max(0., (epoch - config.warmup_epochs) / config.ramp_epochs))
    return {"reconstruction": 1., "cosine": config.cosine_weight,
            "variance": ramp * config.variance_weight,
            "correlation": ramp * config.correlation_weight,
            "relation": ramp * config.relation_weight if config.relation_mode != "none" else 0.}


def encoder_gradient_norms(terms, weights, model):
    result = {}
    parameters = list(model.encoder.parameters())
    for name, term in terms.items():
        if weights[name] == 0:
            result[name] = 0.
            continue
        gradients = torch.autograd.grad(term * weights[name], parameters, retain_graph=True,
                                        allow_unused=True)
        result[name] = float(sum(g.detach().square().sum() for g in gradients if g is not None).sqrt())
    return result
