"""Sample contrastive query positions and match them to distinct next-item keys."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ContrastiveSelection:
    rows: torch.Tensor
    positions: torch.Tensor
    target_ids: torch.Tensor
    available_queries: int

    def __len__(self):
        return len(self.target_ids)


def select_positions(batch, max_queries=None, ignored_label=-100):
    labels = batch["labels"]
    if labels.ndim == 1:
        labels = labels[:, None]
    if labels.ndim != 2 or labels.shape[0] != batch["input_ids"].shape[0]:
        raise ValueError("Labels must have shape [B, T] or [B, 1].")
    attention_mask = batch["attention_mask"].bool()
    valid_labels = (labels != ignored_label) & (labels > 0)
    if labels.shape[1] > 1:
        if labels.shape != attention_mask.shape:
            raise ValueError("Per-position labels must match the history shape.")
        rows, positions = (valid_labels & attention_mask).nonzero(as_tuple=True)
        targets = labels[rows, positions].long()
    else:
        indices = torch.arange(labels.shape[1], device=labels.device)[None, :]
        last_labels = torch.where(valid_labels, indices, -1).max(1).values
        valid_rows = (last_labels >= 0) & (batch["seq_lens"] > 0)
        rows = valid_rows.nonzero(as_tuple=True)[0]
        positions = batch["seq_lens"][rows].long() - 1
        targets = labels[rows, last_labels[rows]].long()
    available = len(targets)
    if max_queries is not None:
        if type(max_queries) is not int or max_queries <= 0:
            raise ValueError("max_queries must be a positive integer.")
        if available > max_queries:
            chosen = torch.randperm(available, device=targets.device)[:max_queries]
            rows, positions, targets = rows[chosen], positions[chosen], targets[chosen]
    return ContrastiveSelection(rows, positions, targets, available)


def item_infonce(queries, keys, target_ids, key_ids, temperature):
    """One column per item ID, so repeated targets are never false negatives."""
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive.")
    if queries.ndim != 2 or keys.ndim != 2 or queries.shape[1] != keys.shape[1]:
        raise ValueError("Queries and keys must share a latent dimension.")
    if len(queries) != len(target_ids) or len(keys) != len(key_ids):
        raise ValueError("Every query and key requires a corresponding item ID.")
    if len(key_ids) != len(key_ids.unique()):
        raise ValueError("Contrastive item keys must have distinct IDs.")
    if not len(queries):
        return queries.sum() * 0 + keys.sum() * 0
    matches = target_ids[:, None] == key_ids[None, :]
    if not matches.any(1).all():
        raise ValueError("Every contrastive target must be present in the item keys.")
    if len(keys) <= 1:
        return queries.sum() * 0 + keys.sum() * 0
    with torch.autocast(device_type=queries.device.type, enabled=False):
        logits = (F.normalize(queries.float(), dim=-1)
                  @ F.normalize(keys.float(), dim=-1).T) / temperature
        return F.cross_entropy(logits, matches.long().argmax(1))


def sample_negative_ids(batch, rows, positive_ids, n_items, count, device):
    """Sample unique catalog IDs per query, excluding the user's full sequence."""
    if count <= 0:
        raise ValueError("Contrastive negative count must be positive.")
    if len(rows) != len(positive_ids):
        raise ValueError("Every contrastive query must have one positive item ID.")
    rows = rows.to(device=device, dtype=torch.long).reshape(-1)
    input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
    input_mask = batch["attention_mask"].to(device=device, dtype=torch.bool)
    labels = batch["labels"].to(device=device, dtype=torch.long)
    if labels.ndim == 1:
        labels = labels[:, None]
    positive_ids = positive_ids.to(device=device, dtype=torch.long).reshape(-1)
    if not len(rows):
        return torch.empty((0, 0), dtype=torch.long, device=device)
    if (rows < 0).any() or (rows >= input_ids.shape[0]).any():
        raise ValueError("Contrastive query rows are outside the batch.")

    # Build one forbidden mask per query, then choose unique IDs with top-k
    # random scores. This avoids thousands of Python-level torch.isin calls.
    forbidden = torch.zeros((len(rows), n_items + 1), dtype=torch.bool, device=device)
    query_inputs = input_ids[rows]
    query_input_mask = input_mask[rows]
    query_labels = labels[rows]
    ids = torch.cat((query_inputs, query_labels), dim=1)
    valid = torch.cat((query_input_mask, query_labels > 0), dim=1)
    ids = ids.clamp_(0, n_items)
    forbidden.scatter_(1, ids, valid)
    forbidden.scatter_(1, positive_ids[:, None].clamp(0, n_items), torch.ones((len(rows), 1),
                                                                               dtype=torch.bool,
                                                                               device=device))
    available = (~forbidden[:, 1:]).sum(dim=1)
    width = min(count, int(available.min().item()))
    if width == 0:
        return torch.empty((len(rows), 0), dtype=torch.long, device=device)
    scores = torch.rand((len(rows), n_items), device=device)
    scores.masked_fill_(forbidden[:, 1:], -torch.inf)
    return scores.topk(width, dim=1, largest=True, sorted=False).indices + 1


def infonce_loss(anchor, positive, negative, loss_mask=None, pos_id=None, neg_id=None,
                 temperature: float = 0.04, **_kwargs):
    """InfoNCE with one positive and a per-query, unique negative set.

    The optional IDs allow callers to mask accidental false negatives. The
    joint trainer already removes every item from the same user's sequence.
    """
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive.")
    if anchor.ndim == 3:
        if positive.shape != anchor.shape or negative.ndim != 3 or negative.shape[0] != anchor.shape[0]:
            raise ValueError("Expected [B,L,D] anchors and [B,K,D] negatives.")
        if loss_mask is None:
            loss_mask = torch.ones(anchor.shape[:2], dtype=torch.bool, device=anchor.device)
        if loss_mask.shape != anchor.shape[:2]:
            raise ValueError("loss_mask must have shape [B,L].")
        rows = loss_mask.nonzero(as_tuple=True)[0]
        flat_pos_id = pos_id[loss_mask] if pos_id is not None else None
        flat_neg_id = neg_id[rows] if neg_id is not None else None
        return infonce_loss(anchor[loss_mask], positive[loss_mask], negative[rows], loss_mask=None,
                            pos_id=flat_pos_id, neg_id=flat_neg_id, temperature=temperature)
    if anchor.ndim != 2 or positive.shape != anchor.shape or negative.ndim != 3:
        raise ValueError("Expected anchor/positive [M,D] and negative [M,K,D].")
    if negative.shape[0] != anchor.shape[0] or negative.shape[2] != anchor.shape[1]:
        raise ValueError("Negative samples must align with anchor dimensions.")
    if loss_mask is None:
        loss_mask = torch.ones(anchor.shape[0], dtype=torch.bool, device=anchor.device)
    if not loss_mask.any():
        return anchor.sum() * 0
    anchor = F.normalize(anchor.float(), dim=-1, eps=1e-6)[loss_mask]
    positive = F.normalize(positive.float(), dim=-1, eps=1e-6)[loss_mask]
    negative = F.normalize(negative.float(), dim=-1, eps=1e-6)[loss_mask]
    positive_logits = (anchor * positive).sum(-1, keepdim=True) / temperature
    negative_logits = torch.einsum("md,mkd->mk", anchor, negative) / temperature
    if pos_id is not None and neg_id is not None:
        false_negative = pos_id[loss_mask, None] == neg_id[loss_mask]
        negative_logits = negative_logits.masked_fill(false_negative, torch.finfo(negative_logits.dtype).min)
    logits = torch.cat((positive_logits, negative_logits), dim=1)
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
