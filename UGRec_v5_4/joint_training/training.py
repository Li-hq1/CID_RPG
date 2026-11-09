"""Alternating AE and recommender updates with explicit gradient routing."""

from contextlib import contextmanager
import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from compression.geometry_ae.model import loss_terms


@contextmanager
def frozen(parameters):
    parameters = list(parameters)
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def scheduler_for(optimizer, settings, total_steps):
    warmup = math.ceil(total_steps * settings.warmup_ratio)

    def factor(step):
        if settings.scheduler == "constant":
            return 1.
        if warmup and step < warmup:
            return step / warmup
        fraction = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1., fraction)))

    return LambdaLR(optimizer, factor)


def gradient_norm(loss, parameters):
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters or not loss.requires_grad:
        return 0.
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squares = [value.detach().square().sum() for value in gradients if value is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.


class JointTrainer:
    def __init__(self, model, config, train_ids, source, total_steps, loader_generator=None):
        self.model, self.config, self.source = model, config, source
        self.train_ids = train_ids.long().cpu()
        if len(self.train_ids) < 2 or not torch.all(self.train_ids > 0):
            raise ValueError("Joint training needs at least two non-padding training items.")
        self.loader_generator = loader_generator or torch.Generator().manual_seed(config.training.seed)
        self.ae_generator = torch.Generator().manual_seed(config.training.seed + 1)
        self.rec_parameters = list(model.rec_parameters())
        self.ae_optimizer = self.ae_scheduler = None
        if config.ae.frozen:
            model.ae.requires_grad_(False)
        else:
            ae_steps = total_steps // config.ae.update_every
            if ae_steps < 1:
                raise ValueError("Training budget is shorter than ae.update_every.")
            self.ae_optimizer = AdamW(model.ae.parameters(), lr=config.ae.lr, weight_decay=config.ae.weight_decay)
            self.ae_scheduler = scheduler_for(self.ae_optimizer, config.ae, ae_steps)
        rec_steps = total_steps // config.recommender.update_every
        if rec_steps < 1:
            raise ValueError("Training budget is shorter than recommender.update_every.")
        self.rec_optimizer = AdamW(self.rec_parameters, lr=config.recommender.lr,
                                  weight_decay=config.recommender.weight_decay)
        self.rec_scheduler = scheduler_for(self.rec_optimizer, config.recommender, rec_steps)
        self.global_step = self.ae_steps = self.rec_steps = 0
        self.diagnostics_reference = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    def sample_items(self, count):
        if self.config.flow_ranking.enabled and count < len(self.train_ids):
            # Exchangeable uniform draws, then a uniform subset of their unique
            # values: exact sampling without replacement, O(count) expected work.
            # Do not allocate a fresh catalog-sized permutation for every AE step.
            positions = torch.empty(0, dtype=torch.long)
            while len(positions) < count:
                extra = torch.randint(len(self.train_ids), (max(2 * count, 16),), generator=self.ae_generator)
                positions = torch.cat((positions, extra)).unique()
            order = torch.randperm(len(positions), generator=self.ae_generator)[:count]
            positions = positions[order]
        else:
            positions = torch.randperm(len(self.train_ids), generator=self.ae_generator)[:count]
        return self.train_ids[positions].to(self.device)

    def step(self, batch, epoch, *, diagnostics=False):
        if self.config.flow_ranking.enabled or (getattr(self.config, "ranking", None) is not None and self.config.ranking.enabled):
            return self._joint_step(batch, diagnostics=diagnostics)
        self.global_step += 1
        metrics = {"ae_updated": 0., "rec_updated": 0.}
        do_ae = self.ae_optimizer is not None and self.global_step % self.config.ae.update_every == 0
        do_rec = self.global_step % self.config.recommender.update_every == 0
        contrastive = self.config.contrastive
        contrastive_weight_value = contrastive.weight if contrastive.enabled else 0.
        weight_ae = contrastive_weight_value if do_ae else 0.
        selection = None
        if weight_ae > 0:
            selection = self.model.select_contrastive_positions(batch)
        self.model.zero_grad(set_to_none=True)
        if do_ae:
            self.model.eval()
            with frozen(self.rec_parameters):
                self.model.ae.train()
                ids = self.sample_items(self.config.ae.batch_size)
                x = self.model.raw_features[ids.cpu() - 1].to(self.device)
                reconstructed, z = self.model.ae(x)
                settings = self.config.ae
                terms = loss_terms(x, reconstructed, z, self.model.ae.input_scale,
                                   settings.relation_temperature, batch_relation=settings.relation_mode == "batch")
                weights = {"reconstruction": 1., "cosine": settings.cosine_weight,
                           "variance": settings.variance_weight, "correlation": settings.correlation_weight,
                           "relation": settings.relation_weight if settings.relation_mode == "batch" else 0.}
                ae_loss = sum(terms[key] * value for key, value in weights.items())
                contrastive_loss = ae_loss.new_zeros(())
                self.model.ae.eval()
                if weight_ae > 0:
                    _, prediction, _ = self.model.rf_loss(
                        batch, return_prediction=True, selection=selection, ae_grad=True
                    )
                    contrastive_loss, info = self.model.contrastive_loss(
                        batch, selection, prediction=prediction
                    )
                    metrics.update({f"contrastive_ae/{key}": float(value) for key, value in info.items()})
                if diagnostics:
                    metrics["ae_grad_norm"] = gradient_norm(ae_loss, self.model.ae.encoder.parameters())
                    metrics["ae_contrastive_grad_norm"] = gradient_norm(weight_ae * contrastive_loss,
                                                                        self.model.ae.encoder.parameters())
                self._backward_step(ae_loss + weight_ae * contrastive_loss, self.ae_optimizer,
                                    self.model.ae.parameters(), settings.max_grad_norm)
                self.ae_scheduler.step()
                self.ae_steps += 1
                self.model.invalidate_catalog()
                metrics.update({f"ae/{key}": float(value.detach()) for key, value in terms.items()})
                metrics.update(ae_loss=float(ae_loss.detach()),
                               ae_contrastive=float(contrastive_loss.detach()), ae_updated=1.)
        # RF must see new AE parameters, with no AE graph or stale gradients.
        self.model.zero_grad(set_to_none=True)
        if do_rec:
            self.model.train()
            self.model.ae.eval()
            rf_loss = self.model.rf_loss(batch)
            self._backward_step(rf_loss, self.rec_optimizer, self.rec_parameters,
                                self.config.recommender.max_grad_norm)
            self.rec_scheduler.step()
            self.rec_steps += 1
            metrics.update(rf_loss=float(rf_loss.detach()), rec_updated=1.)
        metrics.update(contrastive_weight=contrastive_weight_value,
                       contrastive_negative_count=metrics.get("contrastive_ae/negative_count", 0.),
                       ae_lr=self.ae_optimizer.param_groups[0]["lr"] if self.ae_optimizer else 0.,
                       rec_lr=self.rec_optimizer.param_groups[0]["lr"])
        return metrics

    def _joint_step(self, batch, *, diagnostics=False):
        self.global_step += 1
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        objective, terms = self.model.joint_objectives(batch)
        local_weight = (self.config.flow_ranking.local_weight if self.config.flow_ranking.enabled
                        else self.config.ranking.local_weight)
        settings = self.config.ae
        local = objective.new_zeros(())
        if self.ae_optimizer is not None and local_weight > 0:
            ids = self.sample_items(settings.batch_size)
            x = self.model.raw_items(ids)
            reconstructed, z = self.model.ae(x)
            if self.config.flow_ranking.enabled and self.config.flow_ranking.normalized_geometry:
                z = self.model.normalizer(z)
            local_terms = loss_terms(x, reconstructed, z, self.model.ae.input_scale,
                                     settings.relation_temperature, batch_relation=settings.relation_mode == "batch")
            weights = {"reconstruction": 1., "cosine": settings.cosine_weight,
                       "variance": settings.variance_weight, "correlation": settings.correlation_weight,
                       "relation": settings.relation_weight if settings.relation_mode == "batch" else 0.}
            local = sum(local_terms[key] * value for key, value in weights.items())
        loss = objective + local_weight * local
        if not torch.isfinite(loss):
            raise ValueError("Non-finite joint ranking loss.")
        metrics = {f"loss/{key}": float(value.detach()) for key, value in terms.items()}
        if diagnostics and self.ae_optimizer is not None:
            metrics["ae_joint_grad_norm"] = gradient_norm(objective, self.model.ae.encoder.parameters())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rec_parameters, self.config.recommender.max_grad_norm,
                                       error_if_nonfinite=True)
        self.rec_optimizer.step()
        self.rec_scheduler.step()
        self.rec_steps += 1
        if self.ae_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.model.ae.parameters(), settings.max_grad_norm,
                                           error_if_nonfinite=True)
            self.ae_optimizer.step()
            self.ae_scheduler.step()
            self.ae_steps += 1
        self.model.invalidate_catalog()
        metrics.update(ae_loss=float(local.detach()), rf_loss=float(terms.get("cfm", terms.get("rf", local.new_zeros(()))).detach()),
                       loss=float(loss.detach()), ae_updated=float(self.ae_optimizer is not None), rec_updated=1.,
                       ae_lr=self.ae_optimizer.param_groups[0]["lr"] if self.ae_optimizer else 0.,
                       rec_lr=self.rec_optimizer.param_groups[0]["lr"])
        return metrics

    @staticmethod
    def _backward_step(loss, optimizer, parameters, limit):
        if not torch.isfinite(loss):
            raise ValueError("Non-finite joint training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(parameters), limit, error_if_nonfinite=True)
        optimizer.step()

    def state_dict(self):
        result = {"global_step": self.global_step, "ae_steps": self.ae_steps, "rec_steps": self.rec_steps,
                  "loader_rng": self.loader_generator.get_state(), "ae_rng": self.ae_generator.get_state(),
                  "diagnostics_reference": self.diagnostics_reference}
        for key in ("ae_optimizer", "rec_optimizer", "ae_scheduler", "rec_scheduler"):
            value = getattr(self, key)
            result[key] = value.state_dict() if value is not None else None
        return result

    def load_state_dict(self, state):
        self.diagnostics_reference = state.get("diagnostics_reference")
        for key in ("global_step", "ae_steps", "rec_steps"):
            setattr(self, key, int(state[key]))
        for name in ("loader", "ae"):
            getattr(self, f"{name}_generator").set_state(state[f"{name}_rng"].cpu())
        for key in ("ae_optimizer", "rec_optimizer", "ae_scheduler", "rec_scheduler"):
            value = getattr(self, key)
            if value is not None:
                if state[key] is None:
                    raise ValueError(f"Missing {key} state in joint checkpoint.")
                value.load_state_dict(state[key])
