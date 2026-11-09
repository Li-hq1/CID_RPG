"""Matched next CE plus explicit next/future CE or continuous-time CFM."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contrastive import select_positions
from .flow_ranking import _exact_target_logp
from .model import JointModel


def straight_bridge(source, target, fraction):
    source, target, fraction = source.detach(), target.detach(), fraction.detach()
    return (1 - fraction) * target + fraction * source, source - target


def sampled_logp(queries, targets, positive_sets, negative_ids, model, *, keys_grad):
    """Known positive mass exactly once + uniform estimate of remaining mass.

    All catalog items remain in the softmax objective. A known future positive
    drawn by the sampler is accounted for in the exact positive mass, never
    counted again as an independently weighted negative.
    """
    temperature = model.config.ranking.temperature
    queries = F.normalize(queries, dim=-1)
    pos = positive_sets
    keys = F.normalize(model.encode_ids(pos, grad=keys_grad), dim=-1)
    logits_pos = (queries[:, None] * keys).sum(-1) / temperature
    previous = torch.ones(pos.shape[1], pos.shape[1], device=pos.device, dtype=torch.bool).tril(-1)
    duplicates = ((pos[:, :, None] == pos[:, None, :]) & previous).any(-1)
    logits_pos = logits_pos.masked_fill((pos <= 0) | duplicates, -torch.inf)
    negative_keys = F.normalize(model.encode_ids(negative_ids, grad=keys_grad), dim=-1)
    logits_neg = queries @ negative_keys.T / temperature
    known = (negative_ids[None, :, None] == pos[:, None, :]).any(-1)
    logits_neg = logits_neg.masked_fill(known, -torch.inf)
    logits_neg = logits_neg + math.log((model.n_items - 1) / len(negative_ids))
    target_keys = F.normalize(model.encode_ids(targets, grad=keys_grad), dim=-1)
    numerator = (queries * target_keys).sum(-1) / temperature
    return numerator - torch.logsumexp(torch.cat((logits_pos, logits_neg), dim=1), dim=1)


class MultiTargetModel(JointModel):
    def __init__(self, config, raw_features, ae=None):
        super().__init__(config, raw_features, ae)
        if not config.experiment.enabled:
            raise ValueError("The v5_3 research runner needs experiment.enabled=true.")
        # Construct every arm identically, so initialization is matched.
        self.future_head = nn.Linear(self.condition_dim, self.latent_dim)

    @property
    def exact_training(self):
        cfg = self.config.flow_ranking
        return cfg.catalog == "exact" or (cfg.catalog == "auto" and self.n_items - 1 <= cfg.max_exact_items)

    def classification_loss(self, queries, targets, positives, negatives, catalog=None, *, keys_grad):
        if self.exact_training:
            unit_keys = F.normalize(catalog, dim=-1)
            logp = checkpoint(_exact_target_logp, F.normalize(queries, dim=-1), unit_keys,
                             targets - 1, self.config.ranking.temperature, use_reentrant=False)
        else:
            logp = sampled_logp(queries, targets, positives, negatives, self, keys_grad=keys_grad)
        return -logp.mean()

    def joint_objectives(self, batch):
        cfg = self.config.experiment
        selection = select_positions(batch, self.config.flow_ranking.max_queries)
        if not len(selection):
            raise ValueError("Need at least one next-item training label.")
        rows, positions, targets = selection.rows, selection.positions, selection.target_ids
        encoded, _ = self.encode_history(batch, ae_grad=True)
        condition = self.condition_proj(encoded[rows, positions])
        catalog = self.encode_ids(torch.arange(1, self.n_items, device=self.device)) if self.exact_training else None
        # One draw stream for all arms, reused by both CE objectives.
        negatives = torch.randint(1, self.n_items, (self.config.flow_ranking.negatives,), device=self.device)
        ranking = self.classification_loss(self.ranking_head(condition), targets, targets[:, None],
                                           negatives, catalog, keys_grad=True)
        future = batch["future_labels"][rows, positions]
        valid = (future > 0).all(-1)
        n = int(valid.sum())
        auxiliary = ranking.new_zeros(())
        if n:
            # An independent input graph prevents ALL auxiliary AE gradients,
            # while allowing shared HSTU/condition gradients in C and FM.
            aux_encoded, _ = self.encode_history(batch, ae_grad=False)
            aux_condition = self.condition_proj(aux_encoded[rows[valid], positions[valid]])
            future = future[valid]
            index = torch.randint(future.shape[1], (n,), device=self.device)
            future_target = future[torch.arange(n, device=self.device), index]
            noise = torch.randn(n, self.latent_dim, device=self.device)
            fraction = torch.rand(n, 1, device=self.device)
            if cfg.arm == "future_ce":
                auxiliary = self.classification_loss(self.future_head(aux_condition), future_target, future,
                                  negatives, catalog.detach() if catalog is not None else None, keys_grad=False)
            elif cfg.arm in {"next_fm", "future_fm"}:
                target = targets[valid] if cfg.arm == "next_fm" else future_target
                clean = self.encode_ids(target, grad=False)
                base = self.flow_prior(aux_condition, noise).detach()
                state, velocity_label = straight_bridge(base, clean, fraction)
                time = fraction[:, 0] * (self.diffloss.train_diffusion.num_timesteps - 1)
                predicted = self.diffloss.net(state, time, aux_condition)
                auxiliary = F.mse_loss(predicted, velocity_label)
        terms = {"ranking": ranking, "auxiliary": auxiliary}
        self.last_training_stats = {"queries": len(targets), "available_queries": selection.available_queries,
                                    "auxiliary_queries": n, "exact_catalog": int(self.exact_training)}
        return ranking + cfg.auxiliary_weight * auxiliary, terms

    def transport(self, initial, condition, steps):
        """Continuous times, source at 1 and target at 0, with matching sign."""
        state = initial
        grid = torch.linspace(1., 0., steps + 1, device=initial.device)
        for current, following in zip(grid[:-1], grid[1:]):
            time = current.expand(len(initial)) * (self.diffloss.train_diffusion.num_timesteps - 1)
            state = state + (following - current) * self.diffloss.net(state, time, condition)
        return state

    @torch.no_grad()
    def log_probabilities(self, batch, readout=None):
        if self.training or len(self.catalog) != self.n_items:
            raise RuntimeError("Call eval() and refresh_catalog() before scoring.")
        if readout is None:
            readout = {"ce": "ranking", "next_fm": "flow", "future_ce": "future",
                       "future_fm": "flow"}[self.config.experiment.arm]
        _, condition = self.encode_history(batch, ae_grad=False)
        temp = self.config.ranking.temperature
        if readout in {"ranking", "future"}:
            query = self.ranking_head(condition) if readout == "ranking" else self.future_head(condition)
            return (F.normalize(query, dim=-1) @ self.unit_catalog.T / temp).log_softmax(-1)
        if readout not in {"flow", "prior"}:
            raise ValueError("Unknown recommendation readout.")
        draws = self.config.evaluation.eval_num_samples
        expanded = condition.repeat_interleave(draws, 0)
        samples = self.flow_prior(expanded)
        if readout == "flow":
            samples = self.transport(samples, expanded, self.config.experiment.sampling_steps)
        result = samples.new_full((len(condition), self.n_items - 1), -torch.inf)
        chunk = self.config.evaluation.mixture_query_chunk_size
        for start in range(0, len(samples), chunk):
            probabilities = (F.normalize(samples[start:start + chunk], dim=-1) @ self.unit_catalog.T / temp).log_softmax(-1)
            for offset, logp in enumerate(probabilities):
                row = (start + offset) // draws
                result[row] = torch.logaddexp(result[row], logp - math.log(draws))
        return result
