"""Independent composition of an online item AE, history encoder, and RF."""

from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from compression.geometry_ae.model import GeometryAE
from models.diffusion.diffloss import DiffLoss
from models.evaluation_sampling import aggregate_diffusion_candidates, validate_eval_sampling
from models.hstu import HSTUSequenceEncoder
from models.tigerdiff import InputProj, TigerSequenceEncoder

from .contrastive import infonce_loss, select_positions, sample_negative_ids
from .normalization import LatentNormalizer


class JointModel(nn.Module):
    def __init__(self, config, raw_features, ae=None):
        super().__init__()
        self.config = config
        features = torch.as_tensor(raw_features, dtype=torch.float32, device="cpu")
        if features.ndim != 2 or min(features.shape) <= 0 or not torch.isfinite(features).all():
            raise ValueError("Raw features must be a finite, nonempty [items, input_dim] matrix.")
        # This immutable source stays on CPU and is identified, not copied, in checkpoints.
        self.raw_features = features.detach().contiguous()
        self.register_buffer("_feature_cache", None, persistent=False)
        self.n_items = len(features) + 1
        self.latent_dim = config.ae.latent_dim
        self.ae = ae if ae is not None else GeometryAE(features.shape[1], config.ae)
        self.normalizer = LatentNormalizer(self.latent_dim)
        rec = config.recommender
        self.hidden_embd = rec.n_embd
        self.max_seq_len = config.data.max_item_seq_len
        self.input_proj = InputProj(self.latent_dim, self.hidden_embd)
        self.encoder_type = rec.encoder_type
        if self.encoder_type == "transformer":
            self.position_emb = nn.Embedding(self.max_seq_len, self.hidden_embd)
            self.encoder = TigerSequenceEncoder(
                hidden_size=rec.n_embd, num_layers=rec.n_layer, num_heads=rec.n_head,
                inner_size=rec.n_inner,
                activation_name="relu" if rec.activation_function == "relu" else "gelu",
                attn_dropout=rec.attn_pdrop, dropout=rec.embd_pdrop,
                layer_norm_eps=rec.layer_norm_epsilon, causal=True,
            )
        elif self.encoder_type == "hstu":
            self.encoder = HSTUSequenceEncoder(
                embedding_dim=rec.n_embd, num_layers=rec.n_layer, num_heads=rec.n_head,
                hidden_dim=rec.hstu_hidden_dim, attention_dim=rec.hstu_attention_dim,
                max_seq_len=self.max_seq_len, use_group_norm=rec.hstu_use_group_norm,
                output_dropout_ratio=rec.hstu_output_dropout,
                dropout_pr=rec.hstu_attention_dropout,
                attention_scaling=rec.hstu_attention_scaling,
            )
        else:
            raise ValueError("encoder_type must be transformer or hstu.")
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_embd, eps=1e-8),
            nn.Linear(self.hidden_embd, self.hidden_embd),
        )
        self.uncond_condition = nn.Parameter(torch.zeros(self.hidden_embd))
        self.uncond_prob = rec.tiger_uncond_prob
        self.cfg_scale = 1 + rec.tiger_guidance_weight
        self.diff_temperature = rec.diff_temperature
        self.diffusion_batch_mul = rec.diffusion_batch_mul
        self.diffloss = DiffLoss(
            target_channels=self.latent_dim, z_channels=self.hidden_embd,
            width=rec.diffloss_w, depth=rec.diffloss_d,
            num_sampling_steps=rec.num_sampling_steps,
            grad_checkpointing=rec.grad_checkpointing, use_rectified_flow=True,
            rectified_flow_steps=rec.rectified_flow_steps, ode_solver=rec.ode_solver,
        )
        # Compatibility marker for older callers; anchors now come directly
        # from the RF clean-state estimate, so no projection head is needed.
        self.contrastive_head = None
        evaluation = config.evaluation
        self.eval_config = asdict(evaluation) if is_dataclass(evaluation) else vars(evaluation).copy()
        validate_eval_sampling(self.eval_config)
        self.knn_metric = evaluation.tiger_knn_metric
        self.temperature = evaluation.temperature
        if self.knn_metric not in {"l2", "cosine", "inner_product"} or self.temperature <= 0:
            raise ValueError("Invalid evaluation metric or temperature.")
        self.register_buffer("catalog", torch.empty(0, self.latent_dim), persistent=False)
        self.last_generation_stats = None
        ranking = getattr(config, "ranking", None)
        if ranking is not None and ranking.enabled:
            self.ranking_head = nn.Linear(self.hidden_embd, self.latent_dim)
            if ranking.teacher_weight > 0:
                self.teacher_projection = nn.Linear(rec.diffloss_w, ranking.teacher_dim)
                self.register_buffer("teacher", torch.zeros(self.n_items, ranking.teacher_dim))
                self.register_buffer("teacher_ready", torch.tensor(False))

    def set_teacher(self, teacher):
        if not hasattr(self, "teacher") or teacher.shape != self.teacher[1:].shape:
            raise ValueError("Teacher shape/config mismatch.")
        if not torch.isfinite(teacher).all() or (teacher.norm(dim=-1) == 0).any():
            raise ValueError("Teacher must have finite nonzero rows.")
        self.teacher[1:].copy_(F.normalize(teacher.detach().to(self.device), dim=-1))
        self.teacher_ready.fill_(True)

    def cache_features(self):
        """Optional immutable device copy for repeated full-catalog training."""
        self._feature_cache = self.raw_features.to(self.device)

    def raw_items(self, ids):
        if self._feature_cache is not None:
            return self._feature_cache[ids.to(self.device) - 1]
        return self.raw_features[ids.cpu() - 1].to(self.device)

    def ranking_logits(self, condition, catalog):
        return (F.normalize(self.ranking_head(condition), dim=-1)
                @ F.normalize(catalog, dim=-1).T / self.config.ranking.temperature)

    def flow_prior(self, condition, noise=None):
        cfg = self.config.ranking
        if noise is None:
            noise = torch.randn(len(condition), self.latent_dim, device=self.device,
                                dtype=condition.dtype) * self.diff_temperature
        if cfg.prior_mode == "noise":
            return noise
        # The base is learned by history CE, independently of velocity MSE.
        anchor = F.normalize(self.ranking_head(condition), dim=-1) * math.sqrt(self.latent_dim)
        return anchor.detach() + cfg.prior_noise * noise

    def transport(self, initial, condition, steps):
        """Differentiable Euler path, identical to inference for the same start."""
        state = initial
        times = torch.linspace(1., 0., steps + 1, device=state.device, dtype=torch.float32)
        for current, following in zip(times[:-1], times[1:]):
            discrete = torch.round(current * (self.diffloss.train_diffusion.num_timesteps - 1)).long()
            velocity = self.diffloss.net(state, discrete.expand(len(state)), condition)
            state = state + (following - current).to(state.dtype) * velocity
        return state

    @staticmethod
    def rollout_catalog_loss(logits, targets, samples, objective):
        """MC catalog likelihood; marginal CE shares responsibility across draws."""
        if objective == "sample_ce":
            return F.cross_entropy(logits, targets.repeat_interleave(samples) - 1)
        log_probabilities = logits.log_softmax(-1).reshape(len(targets), samples, -1)
        target_logp = log_probabilities.gather(-1, (targets - 1)[:, None, None].expand(-1, samples, 1)).squeeze(-1)
        return -(torch.logsumexp(target_logp, dim=1) - math.log(samples)).mean()

    def joint_objectives(self, batch):
        """History CE updates AE+rec; RF and flow CE never update AE.

        Alignment has a separate live AE input graph and a fixed teacher target.
        No validation labels or teacher-forced values are used at inference.
        """
        cfg = self.config.ranking
        selection = select_positions(batch)
        rows, positions = selection.rows.to(self.device), selection.positions.to(self.device)
        targets = selection.target_ids.to(self.device)
        if not len(targets):
            raise ValueError("Joint objectives need at least one supervised position.")
        terms = {}
        catalog = None
        if cfg.weight > 0 or cfg.prediction_weight > 0 or cfg.endpoint_weight > 0 or cfg.rollout_weight > 0:
            catalog = self.encode_ids(torch.arange(1, self.n_items, device=self.device),
                                      grad=cfg.catalog_ae_grad)
        if cfg.weight > 0:
            encoded, _ = self.encode_history(batch, ae_grad=cfg.history_ae_grad)
            conditions = self.condition_proj(encoded[rows, positions])
            terms["ranking"] = F.cross_entropy(self.ranking_logits(conditions, catalog), targets - 1)
        if cfg.rf_weight > 0 or cfg.prediction_weight > 0 or cfg.endpoint_weight > 0 or cfg.rollout_weight > 0:
            encoded, _ = self.encode_history(batch, ae_grad=False)
            condition = self.condition_proj(encoded[rows, positions])
            if self.training and self.uncond_prob > 0:
                drop = torch.rand(len(condition), device=self.device) < self.uncond_prob
                condition = torch.where(drop[:, None], self.uncond_condition[None], condition)
            clean = self.encode_ids(targets, grad=False)
            times = torch.randint(self.diffloss.train_diffusion.num_timesteps, (len(clean),), device=self.device)
            fraction = times[:, None] / (self.diffloss.train_diffusion.num_timesteps - 1)
            noise = self.flow_prior(condition, torch.randn_like(clean))
            noisy = (1 - fraction) * clean + fraction * noise
            velocity = self.diffloss.net(noisy, times, condition)
            terms["rf"] = F.mse_loss(velocity, noise - clean)
            if cfg.prediction_weight > 0:
                eligible = fraction[:, 0] >= cfg.prediction_min_time
                predicted = noisy - fraction * velocity
                logits = (F.normalize(predicted[eligible], dim=-1)
                          @ F.normalize(catalog.detach(), dim=-1).T / cfg.temperature)
                terms["prediction"] = (F.cross_entropy(logits, targets[eligible] - 1)
                                       if eligible.any() else predicted.sum() * 0)
            if cfg.endpoint_weight > 0:
                # Pure history+noise input: the target enters only as a CE label.
                times_end = torch.full_like(times, self.diffloss.train_diffusion.num_timesteps - 1)
                base = self.flow_prior(condition, torch.randn_like(clean))
                endpoint = base - self.diffloss.net(base, times_end, condition)
                logits = (F.normalize(endpoint, dim=-1) @ F.normalize(catalog.detach(), dim=-1).T / cfg.temperature)
                terms["endpoint"] = F.cross_entropy(logits, targets - 1)
            if cfg.rollout_weight > 0:
                expanded = condition.repeat_interleave(cfg.rollout_samples, 0)
                noise = torch.randn(len(expanded), self.latent_dim, device=clean.device, dtype=clean.dtype)
                base = self.flow_prior(expanded, noise)
                generated = self.transport(base, expanded, cfg.rollout_steps)
                logits = (F.normalize(generated, dim=-1) @ F.normalize(catalog.detach(), dim=-1).T / cfg.temperature)
                terms["rollout"] = self.rollout_catalog_loss(logits, targets, cfg.rollout_samples, cfg.rollout_objective)
        if cfg.teacher_weight > 0:
            if not self.teacher_ready.item():
                raise ValueError("Set a fixed train-only teacher before alignment training.")
            encoded, _ = self.encode_history(batch, ae_grad=cfg.teacher_ae_grad)
            condition = self.condition_proj(encoded[rows, positions])
            clean = self.encode_ids(targets, grad=cfg.teacher_ae_grad)
            minimum = math.ceil(cfg.teacher_min_time * (self.diffloss.train_diffusion.num_timesteps - 1))
            times = torch.randint(minimum, self.diffloss.train_diffusion.num_timesteps,
                                  (len(clean),), device=self.device)
            fraction = times[:, None] / (self.diffloss.train_diffusion.num_timesteps - 1)
            noisy = (1 - fraction) * clean + fraction * self.flow_prior(condition, torch.randn_like(clean))
            _, hidden = self.diffloss.net(noisy, times, condition, return_features=True)
            terms["teacher"] = (1 - F.cosine_similarity(self.teacher_projection(hidden),
                                                       self.teacher[targets], dim=-1)).mean()
        weights = {"ranking": cfg.weight, "rf": cfg.rf_weight,
                   "prediction": cfg.prediction_weight, "endpoint": cfg.endpoint_weight,
                   "rollout": cfg.rollout_weight,
                   "teacher": cfg.teacher_weight}
        return sum(weights[key] * value for key, value in terms.items()), terms

    @property
    def device(self):
        return next(self.ae.parameters()).device

    def rec_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith(("ae.", "normalizer.")):
                yield parameter

    def encode_ids(self, ids, grad=True):
        ids = ids.to(device=self.device, dtype=torch.long)
        if ids.numel() and ((ids < 0).any() or (ids >= self.n_items).any()):
            raise ValueError("Item ID is outside the catalog.")
        unique_ids, inverse = ids.reshape(-1).unique(sorted=True, return_inverse=True)
        positive = unique_ids > 0
        with nullcontext() if grad else torch.no_grad():
            values = self.raw_items(unique_ids[positive])
            if len(values):
                latent = self.normalizer(self.ae.latent(values))
                lookup = latent.new_zeros((len(unique_ids), self.latent_dim))
                lookup = lookup.index_copy(0, positive.nonzero(as_tuple=True)[0], latent)
            else:
                lookup = self.raw_features.new_zeros((len(unique_ids), self.latent_dim), device=self.device)
            return lookup[inverse].reshape(*ids.shape, self.latent_dim)

    def encode_history(self, batch, ae_grad=True):
        ids = batch["input_ids"]
        mask = batch["attention_mask"].to(device=self.device, dtype=torch.bool)
        if ids.ndim != 2 or mask.shape != ids.shape or not 0 < ids.shape[1] <= self.max_seq_len:
            raise ValueError("History IDs and masks must have shape [B, T] within max_item_seq_len.")
        lengths = batch["seq_lens"].to(device=self.device, dtype=torch.long)
        if lengths.shape != (ids.shape[0],) or (lengths <= 0).any() or (lengths > ids.shape[1]).any():
            raise ValueError("Every history requires a valid, positive sequence length.")
        if not torch.equal(mask.sum(1), lengths):
            raise ValueError("Sequence lengths must agree with the attention mask.")
        inputs = self.input_proj(self.encode_ids(ids, grad=ae_grad))
        if self.encoder_type == "transformer":
            positions = torch.arange(ids.shape[1], device=self.device)[None, :]
            inputs = inputs + self.position_emb(positions)
        encoded = self.encoder(inputs, attention_mask=mask)
        rows = torch.arange(len(ids), device=self.device)
        condition = self.condition_proj(encoded[rows, lengths - 1])
        return encoded, condition

    def select_contrastive_positions(self, batch):
        return select_positions(batch, self.config.contrastive.max_queries)

    def _rf_forward(self, batch, selection=None, ae_grad=False):
        encoded, condition = self.encode_history(batch, ae_grad=ae_grad)
        selection = select_positions(batch) if selection is None else selection
        if not len(selection):
            zero = condition.sum() * 0 + next(self.diffloss.parameters()).sum() * 0
            return zero, condition.new_empty((0, self.latent_dim)), selection
        rows, positions = selection.rows.to(self.device), selection.positions.to(self.device)
        c = self.condition_proj(encoded[rows, positions])
        target = self.encode_ids(selection.target_ids, grad=False)
        if self.training and self.uncond_prob > 0:
            uncond_mask = torch.rand(len(c), device=self.device) < self.uncond_prob
            c = torch.where(uncond_mask[:, None], self.uncond_condition[None, :].to(c.dtype), c)
        if self.diffusion_batch_mul > 1:
            c = c.repeat(self.diffusion_batch_mul, 1)
            target = target.repeat(self.diffusion_batch_mul, 1)
        loss, prediction = self.diffloss(target=target, z=c, return_prediction=True)
        return loss, prediction[:len(selection)], selection

    def rf_loss(self, batch, *, return_prediction=False, selection=None, ae_grad=False):
        loss, prediction, supervision = self._rf_forward(batch, selection=selection, ae_grad=ae_grad)
        return (loss, prediction, supervision) if return_prediction else loss

    def contrastive_loss(self, batch, selection=None, prediction=None):
        settings = self.config.contrastive
        if not settings.enabled:
            return torch.zeros((), device=self.device), {"queries": 0, "unique_keys": 0,
                      "negative_count": 0, "available_queries": 0}
        selection = self.select_contrastive_positions(batch) if selection is None else selection
        if not len(selection):
            zero = next(self.parameters()).sum() * 0
            return zero, {"queries": 0, "unique_keys": 0, "negative_count": 0,
                          "available_queries": selection.available_queries}
        if prediction is None:
            _, prediction, _ = self.rf_loss(batch, return_prediction=True, selection=selection, ae_grad=True)
        target_ids = selection.target_ids.to(self.device)
        rows = selection.rows.to(self.device)
        negative_ids = sample_negative_ids(batch, rows, target_ids, self.n_items - 1,
                                           getattr(settings, "num_negatives", 64), self.device)
        if negative_ids.shape[1] == 0:
            zero = prediction.sum() * 0
            return zero, {"queries": len(selection), "unique_keys": int(target_ids.unique().numel()),
                          "negative_count": 0,
                          "available_queries": selection.available_queries}
        positive = self.encode_ids(target_ids)
        negative = self.encode_ids(negative_ids)
        loss = infonce_loss(prediction, positive, negative, temperature=settings.temperature)
        return loss, {"queries": len(selection), "unique_keys": int(target_ids.unique().numel()),
                      "negative_count": negative_ids.shape[1],
                      "available_queries": selection.available_queries}

    def invalidate_catalog(self):
        self.catalog = self.catalog.new_empty((0, self.latent_dim))

    @torch.no_grad()
    def refresh_catalog(self, batch_size=512):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Catalog batch size must be positive.")
        states = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            catalog = torch.zeros(self.n_items, self.latent_dim, device=self.device)
            for start in range(1, self.n_items, batch_size):
                ids = torch.arange(start, min(start + batch_size, self.n_items), device=self.device)
                catalog[ids] = self.encode_ids(ids, grad=False).float()
            if not torch.isfinite(catalog).all():
                raise ValueError("Catalog contains non-finite latent representations.")
            self.catalog = catalog
        finally:
            for module, state in states:
                module.training = state
        return self.catalog

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.invalidate_catalog()
        return result

    def _sample_condition(self, condition):
        ranking = getattr(self.config, "ranking", None)
        if getattr(self.config.evaluation, "prediction_source", "flow") == "prior":
            return self.flow_prior(condition)
        if getattr(self.config.evaluation, "prediction_source", "flow") == "one_step":
            base = (self.flow_prior(condition) if ranking is not None and ranking.enabled
                    else torch.randn(len(condition), self.latent_dim, device=self.device) * self.diff_temperature)
            time = self.diffloss.train_diffusion.num_timesteps - 1
            times = torch.full((len(base),), time, device=self.device, dtype=torch.long)
            if self.cfg_scale != 1:
                c = torch.cat((condition, self.uncond_condition[None].expand_as(condition)))
                velocity = self.diffloss.net.forward_with_cfg(torch.cat((base, base)),
                           torch.cat((times, times)), c, self.cfg_scale)[:len(base)]
            else:
                velocity = self.diffloss.net(base, times, condition)
            return base - velocity
        if ranking is not None and ranking.enabled and ranking.prior_mode == "history":
            return self.diffloss.sample(condition, cfg=1., initial_noise=self.flow_prior(condition))
        c = condition
        if self.cfg_scale != 1:
            c = torch.cat((c, self.uncond_condition[None, :].to(c.dtype).expand_as(c)))
        return self.diffloss.sample(c, temperature=self.diff_temperature,
                                    cfg=self.cfg_scale)[:len(condition)]

    @torch.no_grad()
    def generate(self, batch, n_return_sequences=1):
        if self.training:
            raise RuntimeError("Call eval() before recommendation generation.")
        if type(n_return_sequences) is not int or n_return_sequences <= 0:
            raise ValueError("n_return_sequences must be positive.")
        if len(self.catalog) != self.n_items:
            raise RuntimeError("Refresh the latent catalog before evaluation.")
        self.last_generation_stats = None
        _, condition = self.encode_history(batch, ae_grad=False)
        items = self.catalog[1:]
        if getattr(self.config.evaluation, "prediction_source", "flow") == "ranking":
            return (self.ranking_logits(condition, items).topk(min(n_return_sequences, len(items))).indices + 1).unsqueeze(-1)
        if self.eval_config.get("eval_sampling_mode", "knn") == "diffusion_multi":
            count = self.eval_config.get("eval_num_samples", 8)
            expanded = condition.repeat_interleave(count, 0)
            samples = (self.flow_prior(expanded) if getattr(self.config.evaluation, "prediction_source", "flow") == "prior"
                       else self._sample_condition(expanded))
            samples = samples.reshape(len(condition), count, self.latent_dim)
            predictions, statistics = aggregate_diffusion_candidates(
                samples, items, n_return_sequences, self.eval_config,
            )
            if self.eval_config.get("eval_log_sampling_stats", False):
                self.last_generation_stats = statistics
            return predictions
        samples = self._sample_condition(condition).float()
        if self.knn_metric == "l2":
            scores = -torch.cdist(samples, items)
        elif self.knn_metric == "cosine":
            scores = F.normalize(samples, dim=-1) @ F.normalize(items, dim=-1).T / self.temperature
        else:
            scores = samples @ items.T / self.temperature
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite candidate retrieval scores.")
        k = min(n_return_sequences, len(items))
        return (scores.topk(k, dim=-1).indices + 1).unsqueeze(-1)
