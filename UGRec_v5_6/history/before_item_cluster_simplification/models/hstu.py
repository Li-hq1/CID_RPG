"""Causal HSTU adapted from GenFusion_v4 for item-history encoding."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def reverse_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Right-padded histories count down to 1; padding has position 0."""
    lengths = attention_mask.long().sum(dim=1, keepdim=True)
    positions = lengths - torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return positions.clamp_min(0).masked_fill(~attention_mask.bool(), 0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape_as(x)


class SimpleRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("hstu_attention_dim must be positive and even for RoPE.")
        self.register_buffer(
            "inv_freq", base ** (-torch.arange(dim // 2).float() / (dim // 2)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # Compute phases in FP32 even when the encoder runs under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            phases = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
            phases = phases.repeat_interleave(2, dim=-1)
            cos = phases.cos().unsqueeze(1).to(x.dtype)
            sin = phases.sin().unsqueeze(1).to(x.dtype)
        return cos, sin


def apply_rotary_pos_emb(query, key, cos, sin):
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


class HSTULayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, attention_dim,
                 use_group_norm=True, output_dropout_ratio=0.1, dropout_pr=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.use_group_norm = use_group_norm
        self.output_dropout_ratio = output_dropout_ratio
        self.dropout_pr = dropout_pr
        self.attn_alpha = 1.0 / math.sqrt(attention_dim)
        projection_dim = 2 * num_heads * (hidden_dim + attention_dim)
        self.uvqk_weight = nn.Parameter(torch.empty(embedding_dim, projection_dim))
        self.uvqk_bias = nn.Parameter(torch.zeros(projection_dim))
        self.input_norm_weight = nn.Parameter(torch.ones(embedding_dim))
        self.input_norm_bias = nn.Parameter(torch.zeros(embedding_dim))
        self.output_weight = nn.Parameter(torch.empty(3 * num_heads * hidden_dim, embedding_dim))
        norm_dim = num_heads if use_group_norm else num_heads * hidden_dim
        self.output_norm_weight = nn.Parameter(torch.ones(norm_dim))
        self.output_norm_bias = nn.Parameter(torch.zeros(norm_dim))
        self.rope = SimpleRotaryEmbedding(attention_dim)
        nn.init.xavier_uniform_(self.uvqk_weight)
        nn.init.xavier_uniform_(self.output_weight)

    def forward(self, x, valid_attn_mask, position_ids, denominator,
                return_attentions=False):
        batch_size, seq_len, embedding_dim = x.shape
        flat_x = x.reshape(-1, embedding_dim)
        normed = F.layer_norm(
            flat_x, (embedding_dim,), self.input_norm_weight, self.input_norm_bias, 1e-6,
        )
        uvqk = torch.addmm(self.uvqk_bias, normed, self.uvqk_weight)
        value_width = self.num_heads * self.hidden_dim
        attention_width = self.num_heads * self.attention_dim
        u, v, q, k = uvqk.split(
            [value_width, value_width, attention_width, attention_width], dim=-1,
        )
        u = F.silu(u)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        cos, sin = self.rope(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        weights = F.silu(torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha) / denominator
        weights = weights.masked_fill(~valid_attn_mask, 0.0)
        weights = F.dropout(weights, p=self.dropout_pr, training=self.training)
        attn = torch.matmul(weights, v).transpose(1, 2).reshape(-1, value_width)
        if self.use_group_norm:
            normed_attn = F.group_norm(
                attn.reshape(-1, self.num_heads, self.hidden_dim), self.num_heads,
                self.output_norm_weight, self.output_norm_bias, 1e-6,
            ).reshape(-1, value_width)
        else:
            normed_attn = F.layer_norm(
                attn, (value_width,), self.output_norm_weight, self.output_norm_bias, 1e-6,
            )
        # Preserve the source's raw attention branch as well as the normalized gate.
        y = torch.cat([u, attn, u * normed_attn], dim=-1)
        y = F.dropout(y, p=self.output_dropout_ratio, training=self.training).to(attn.dtype)
        output = torch.addmm(flat_x, y, self.output_weight.to(attn.dtype)).reshape_as(x)
        if return_attentions:
            return output, weights
        return output


class HSTUSequenceEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, attention_dim, num_layers,
                 max_seq_len, use_group_norm=True, output_dropout_ratio=0.1,
                 dropout_pr=0.0, attention_scaling="input_length"):
        super().__init__()
        for name, value in (("embedding_dim", embedding_dim), ("num_heads", num_heads),
                            ("hstu_hidden_dim", hidden_dim), ("hstu_attention_dim", attention_dim),
                            ("num_layers", num_layers), ("max_seq_len", max_seq_len)):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if attention_dim % 2:
            raise ValueError("hstu_attention_dim must be even for RoPE.")
        for name, value in (("hstu_output_dropout", output_dropout_ratio),
                            ("hstu_attention_dropout", dropout_pr)):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")
        if attention_scaling not in {"input_length", "max_seq_len"}:
            raise ValueError("hstu_attention_scaling must be input_length or max_seq_len.")
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.attention_scaling = attention_scaling
        self.causal = True
        self.layers = nn.ModuleList([
            HSTULayer(embedding_dim, num_heads, hidden_dim, attention_dim,
                      use_group_norm, output_dropout_ratio, dropout_pr)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, attention_mask, return_attentions=False, *, item_group_size=1):
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.embedding_dim:
            raise ValueError("hidden_states must have shape [B, T, embedding_dim].")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must have shape [B, T].")
        seq_len = hidden_states.shape[1]
        if type(item_group_size) is not int or item_group_size <= 0 or seq_len % item_group_size:
            raise ValueError("item_group_size must be positive and divide token length.")
        item_length = seq_len // item_group_size
        if not 0 < item_length <= self.max_seq_len:
            raise ValueError("History length must be between 1 and max_seq_len.")
        attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        grouped_mask = attention_mask.reshape(len(hidden_states), item_length, item_group_size)
        if item_group_size > 1 and not torch.equal(grouped_mask, grouped_mask[..., :1].expand_as(grouped_mask)):
            raise ValueError("All cluster tokens of one item must have the same validity.")
        # Cluster order has no temporal meaning. All K tokens of an item share
        # its original RoPE phase and can attend to the whole current-item block.
        position_ids = reverse_position_ids(grouped_mask[..., 0]).repeat_interleave(item_group_size, dim=1)
        item_index = torch.arange(seq_len, device=hidden_states.device) // item_group_size
        causal_mask = item_index[:, None] >= item_index[None, :]
        valid_attn_mask = (
            causal_mask[None, None] & attention_mask[:, None, None, :]
            & attention_mask[:, None, :, None]
        )
        denominator = seq_len if self.attention_scaling == "input_length" else self.max_seq_len * item_group_size
        padding_mask = ~attention_mask.unsqueeze(-1)
        hidden_states = hidden_states.masked_fill(padding_mask, 0.0)
        attentions = []
        for layer in self.layers:
            result = layer(hidden_states, valid_attn_mask, position_ids, denominator, return_attentions)
            if return_attentions:
                hidden_states, weights = result
                attentions.append(weights)
            else:
                hidden_states = result
            hidden_states = hidden_states.masked_fill(padding_mask, 0.0)
        if return_attentions:
            return hidden_states, attentions
        return hidden_states
