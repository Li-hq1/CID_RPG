"""History CE and straight-bridge CFM with explicit, testable gradient routes."""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contrastive import select_positions
from .flow_ranking import candidate_ids, _exact_target_logp, _sampled_target_logp
from .model import JointModel


class DecoupledFMModel(JointModel):
    def __init__(self, config, raw_features, ae=None):
        super().__init__(config, raw_features, ae)
        self.fm_config = config.fm
        self.ranking_head = nn.Linear(self.hidden_embd, self.latent_dim)
        self.ranking_tail = self.fm_tail = self.fm_condition_proj = None
        if config.fm.architecture == "parallel":
            # Reuse the original ranking path exactly; clone only its tail.
            # Branches start equal but own physically separate parameters.
            depth = config.fm.tail_layers
            self.ranking_tail = deepcopy(self.encoder)
            self.ranking_tail.layers = nn.ModuleList(list(self.encoder.layers)[-depth:])
            self.fm_tail = deepcopy(self.ranking_tail)
            self.encoder.layers = nn.ModuleList(list(self.encoder.layers)[:-depth])
            self.fm_condition_proj = deepcopy(self.condition_proj)

    def encode_history(self, batch, ae_grad=True):
        """Public history interface returns the complete ranking HSTU path."""
        states, _ = super().encode_history(batch, ae_grad=ae_grad)
        if self.ranking_tail is not None:
            mask = batch["attention_mask"].to(self.device, dtype=torch.bool)
            states = self.ranking_tail(states, mask)
        rows = torch.arange(len(states), device=self.device)
        lengths = batch["seq_lens"].to(self.device)
        return states, self.condition_proj(states[rows, lengths - 1])

    def history_branches(self, batch):
        """Return all-position CE and FM conditions; labels never enter this path."""
        shared, _ = super().encode_history(batch, ae_grad=True)
        mask = batch["attention_mask"].to(self.device, dtype=torch.bool)
        rank_states = (self.ranking_tail(shared, mask)
                       if self.ranking_tail is not None else shared)
        rank_condition = self.condition_proj(rank_states)
        scope = self.fm_config.mse_scope
        if scope == "shared":
            # A second graph is necessary: allow HSTU/input_proj gradients while
            # excluding AE gradients. Detaching the first graph would block both.
            fm_shared, _ = super().encode_history(batch, ae_grad=False)
        else:
            fm_shared = shared.detach()
        if self.fm_tail is not None:
            fm_condition = self.fm_condition_proj(self.fm_tail(fm_shared, mask))
        elif scope == "shared":
            fm_condition = self.condition_proj(fm_shared)
        else:
            fm_condition = rank_condition.detach()
        if scope == "flow":
            fm_condition = fm_condition.detach()
        return rank_condition, fm_condition

    def flow_prior(self, rank_condition, noise=None):
        """Supervised history anchor plus full-rank noise, fixed for CFM."""
        anchor = F.normalize(self.ranking_head(rank_condition), dim=-1) * math.sqrt(self.latent_dim)
        if noise is None:
            noise = torch.randn_like(anchor)
        return anchor.detach() + self.fm_config.noise_scale * noise

    @staticmethod
    def bridge(clean, base, fraction):
        """Reverse-time convention: t=1 is source; t=0 is the item latent."""
        clean, base = clean.detach(), base.detach()
        return (1 - fraction) * clean + fraction * base, base - clean

    def joint_objectives(self, batch):
        cfg = self.fm_config
        selection = select_positions(batch, cfg.max_queries)
        if not len(selection):
            raise ValueError("Decoupled FM needs a supervised history position.")
        rows, positions = selection.rows.to(self.device), selection.positions.to(self.device)
        targets = selection.target_ids.to(self.device)
        rank_all, fm_all = self.history_branches(batch)
        rank_condition, fm_condition = rank_all[rows, positions], fm_all[rows, positions]
        exact = cfg.catalog == "exact" or (cfg.catalog == "auto" and self.n_items - 1 <= cfg.max_exact_items)
        if exact:
            ids = torch.arange(1, self.n_items, device=self.device)
            labels, negatives = targets - 1, None
        else:
            ids, labels, negatives = candidate_ids(targets, self.n_items, cfg.negatives, return_negative=True)
        keys = self.encode_ids(ids, grad=True)
        queries = F.normalize(self.ranking_head(rank_condition), dim=-1)
        unit_keys = F.normalize(keys, dim=-1)
        if exact:
            logp = checkpoint(_exact_target_logp, queries, unit_keys, labels, cfg.temperature,
                             use_reentrant=False)
        else:
            logp = _sampled_target_logp(queries, unit_keys, labels, negatives, cfg.temperature,
                                        self.n_items - 1)
        ce = -logp.mean()
        clean = keys[labels].detach().repeat_interleave(cfg.train_draws, 0)
        condition = fm_condition.repeat_interleave(cfg.train_draws, 0)
        base = self.flow_prior(rank_condition.repeat_interleave(cfg.train_draws, 0))
        fraction = torch.rand(len(clean), 1, device=self.device)
        state, target_velocity = self.bridge(clean, base, fraction)
        times = fraction[:, 0] * (self.diffloss.train_diffusion.num_timesteps - 1)
        velocity = self.diffloss.net(state, times, condition)
        mse = F.mse_loss(velocity, target_velocity)
        self.last_training_stats = {
            "queries": len(targets), "available_queries": selection.available_queries,
            "candidates": len(keys), "exact_ce": int(exact), "fm_queries": len(clean),
            "time_mean": float(fraction.mean()),
            "target_velocity_rms": float(target_velocity.square().mean().sqrt()),
            "predicted_velocity_rms": float(velocity.detach().square().mean().sqrt()),
        }
        return cfg.ce_weight * ce + cfg.mse_weight * mse, {"history_ce": ce, "rf": mse}

    def transport(self, initial, condition, steps):
        """Continuous-time Euler integration, using the same time scale as CFM."""
        if type(steps) is not int or steps <= 0:
            raise ValueError("Euler steps must be a positive integer.")
        state = initial
        for index in range(steps):
            fraction = 1 - index / steps
            times = state.new_full((len(state),), fraction * (self.diffloss.train_diffusion.num_timesteps - 1))
            state = state - self.diffloss.net(state, times, condition) / steps
        return state

    @torch.no_grad()
    def mixture_log_probabilities(self, batch, draws=None, zero=False, source=None):
        if self.training:
            raise RuntimeError("Call eval() before recommendation generation.")
        if len(self.catalog) != self.n_items:
            raise RuntimeError("Refresh the latent catalog before evaluation.")
        cfg = self.fm_config
        source = source or self.config.evaluation.prediction_source
        rank_all, fm_all = self.history_branches(batch)
        lengths = batch["seq_lens"].to(self.device)
        rows = torch.arange(len(lengths), device=self.device)
        rank_condition, condition = rank_all[rows, lengths - 1], fm_all[rows, lengths - 1]
        count = draws if draws is not None else self.config.evaluation.eval_num_samples
        if type(count) is not int or count <= 0:
            raise ValueError("FM draws must be a positive integer.")
        if source not in {"ranking", "prior", "one_step", "flow"}:
            raise ValueError(f"Unknown FM prediction source: {source}")
        if source == "ranking":
            count = 1
            samples = self.ranking_head(rank_condition)
        else:
            if zero:
                count = 1
            rank_expanded = rank_condition.repeat_interleave(count, 0)
            expanded = condition.repeat_interleave(count, 0)
            noise = (expanded.new_zeros((len(expanded), self.latent_dim)) if zero else None)
            samples = self.flow_prior(rank_expanded, noise)
            if source != "prior":
                samples = self.transport(samples, expanded, 1 if source == "one_step" else cfg.sampling_steps)
        b = len(lengths)
        queries = F.normalize(samples, dim=-1).reshape(b, count, self.latent_dim)
        keys = F.normalize(self.catalog[1:], dim=-1)
        # Chunk only the sample axis; every softmax still sees the entire catalog.
        total = None
        for start in range(0, count, 8):
            logits = queries[:, start:start + 8] @ keys.T / cfg.temperature
            partial = torch.logsumexp(logits.log_softmax(-1), dim=1)
            total = partial if total is None else torch.logaddexp(total, partial)
        self.last_generation_stats = {
            "draws": count, "velocity_nfe": (0 if source in {"ranking", "prior"} else
                                            1 if source == "one_step" else cfg.sampling_steps),
            "source": source,
        }
        return total - math.log(count)


def build_model(config, raw_features, ae=None):
    cls = DecoupledFMModel if config.fm.enabled else JointModel
    return cls(config, raw_features, ae)
