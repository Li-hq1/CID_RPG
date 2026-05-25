# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.models.diffusion.diffloss import DiffLoss
from genrec.tokenizer import AbstractTokenizer


class InputProj(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


def _activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    return nn.GELU()


class TigerSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_dropout: float,
        resid_dropout: float,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        key_mask = attention_mask[:, None, None, :].bool()
        attn_scores = attn_scores.masked_fill(
            ~key_mask,
            torch.finfo(attn_scores.dtype).min,
        )
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = attn_probs.masked_fill(~key_mask, 0.0)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        context = self.out_proj(context)
        return self.resid_dropout(context)


class TigerSequenceEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        inner_size: int,
        activation_name: str,
        attn_dropout: float,
        dropout: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.self_attn = TigerSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            _activation_module(activation_name),
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.attn_norm(hidden_states),
            attention_mask,
        )
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)


class TigerSequenceEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        inner_size: int,
        activation_name: str,
        attn_dropout: float,
        dropout: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TigerSequenceEncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    activation_name=activation_name,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)


class TigerDiff(AbstractModel):
    def __init__(
        self, config: dict, dataset: AbstractDataset, tokenizer: AbstractTokenizer
    ):
        super(TigerDiff, self).__init__(config, dataset, tokenizer)

        self.register_buffer("item_id2embs", self._map_item_embs())

        self.feat_embd = self.item_id2embs.shape[1]
        self.hidden_embd = self.config["n_embd"]
        self.input_proj = InputProj(self.feat_embd, self.hidden_embd)
        self.position_emb = nn.Embedding(
            tokenizer.max_token_seq_len, self.hidden_embd
        )

        self.encoder = TigerSequenceEncoder(
            hidden_size=self.hidden_embd,
            num_layers=self.config["n_layer"],
            num_heads=self.config["n_head"],
            inner_size=self.config["n_inner"],
            activation_name=self._encoder_activation(
                self.config.get("activation_function", "gelu")
            ),
            attn_dropout=self.config.get("attn_pdrop", 0.1),
            dropout=self.config.get("embd_pdrop", 0.1),
            layer_norm_eps=self.config.get("layer_norm_epsilon", 1e-12),
        )
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_embd, eps=1e-8),
            nn.Linear(self.hidden_embd, self.hidden_embd),
        )

        self.temperature = float(self.config.get("temperature", 1.0))
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.knn_metric = self.config.get("tiger_knn_metric", "inner_product")
        if self.knn_metric not in {"inner_product", "cosine", "l2"}:
            raise ValueError(
                "tiger_knn_metric must be one of: inner_product, cosine, l2."
            )

        self.diffusion_batch_mul = int(self.config.get("diffusion_batch_mul", 1))
        self.diff_temperature = float(self.config.get("diff_temperature", 1.0))
        guidance_weight = self.config.get(
            "tiger_guidance_weight",
            self.config.get("tiger_guidance_strength", 0.0),
        )
        self.guidance_weight = float(guidance_weight)
        self.cfg_scale = 1.0 + self.guidance_weight
        self.uncond_prob = float(self.config.get("tiger_uncond_prob", 0.1))
        if not 0.0 <= self.uncond_prob <= 1.0:
            raise ValueError("tiger_uncond_prob must be in [0, 1].")
        self.uncond_condition = nn.Parameter(torch.zeros(self.hidden_embd))

        self.diffloss = DiffLoss(
            target_channels=self.feat_embd,
            z_channels=self.hidden_embd,
            width=self.config.get("diffloss_w", 1024),
            depth=self.config.get("diffloss_d", 3),
            num_sampling_steps=self.config.get("num_sampling_steps", 100),
            grad_checkpointing=self.config.get("grad_checkpointing", False),
            use_rectified_flow=self.config.get("use_rectified_flow", False),
            rectified_flow_steps=self.config.get("rectified_flow_steps", 1000),
            ode_solver=self.config.get("ode_solver", "euler"),
        )

    @staticmethod
    def _encoder_activation(name: str) -> str:
        if name in {"gelu", "gelu_new", "gelu_fast"}:
            return "gelu"
        if name == "relu":
            return "relu"
        return "gelu"

    def _map_item_embs(self) -> torch.Tensor:
        sent_embs = torch.as_tensor(self.tokenizer.sent_embs, dtype=torch.float32)
        pad_embs = torch.zeros((1, sent_embs.shape[1]), dtype=torch.float32)
        return torch.cat([pad_embs, sent_embs], dim=0)

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        diffusion_params = sum(
            p.numel() for p in self.diffloss.parameters() if p.requires_grad
        )
        return (
            f"#Diffusion parameters: {diffusion_params}\n"
            f"#Non-diffusion parameters: {total_params - diffusion_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def _encode_history(self, batch: dict):
        input_embs = self.item_id2embs[batch["input_ids"]]
        input_embs = self.input_proj(input_embs)

        seq_len = input_embs.shape[1]
        positions = torch.arange(seq_len, device=input_embs.device).unsqueeze(0)
        input_embs = input_embs + self.position_emb(positions)

        attention_mask = batch["attention_mask"].bool()
        encoded = self.encoder(
            input_embs,
            attention_mask=attention_mask,
        )

        last_pos = (batch["seq_lens"] - 1).clamp_min(0)
        condition = encoded.gather(
            dim=1,
            index=last_pos.view(-1, 1, 1).expand(-1, 1, self.hidden_embd),
        ).squeeze(1)
        condition = self.condition_proj(condition)
        return encoded, condition

    def _last_valid_labels(self, labels: torch.Tensor):
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        label_mask = labels != self.tokenizer.ignored_label
        positions = torch.arange(labels.shape[1], device=labels.device).view(1, -1)
        last_positions = torch.where(label_mask, positions, -1).max(dim=1).values
        valid_rows = last_positions >= 0

        target_ids = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)
        if valid_rows.any():
            row_ids = valid_rows.nonzero(as_tuple=True)[0]
            target_ids[row_ids] = labels[row_ids, last_positions[valid_rows]].long()
        return target_ids, valid_rows

    def _item_similarity_logits(self, pred_embs: torch.Tensor) -> torch.Tensor:
        item_embs = self.item_id2embs[1:]
        if self.knn_metric == "cosine":
            item_embs = F.normalize(item_embs, p=2, dim=-1)
            pred_embs = F.normalize(pred_embs, p=2, dim=-1)
            return torch.matmul(pred_embs, item_embs.T) / self.temperature
        if self.knn_metric == "l2":
            return -torch.cdist(pred_embs, item_embs, p=2)
        return torch.matmul(pred_embs, item_embs.T) / self.temperature

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        encoded, condition = self._encode_history(batch)
        outputs = SimpleNamespace()
        outputs.last_hidden_state = encoded
        outputs.final_states = condition

        if not return_loss:
            return outputs

        assert "labels" in batch, "The batch must contain the labels."
        target_ids, valid_rows = self._last_valid_labels(batch["labels"])
        if not valid_rows.any():
            outputs.loss = condition.sum() * 0.0
            return outputs

        z = condition[valid_rows]
        target = self.item_id2embs[target_ids[valid_rows]]

        if self.training and self.uncond_prob > 0.0:
            uncond_mask = torch.rand(z.shape[0], device=z.device) < self.uncond_prob
            if uncond_mask.any():
                z = z.clone()
                z[uncond_mask] = self.uncond_condition.to(dtype=z.dtype)

        if self.diffusion_batch_mul > 1:
            z = z.repeat(self.diffusion_batch_mul, 1)
            target = target.repeat(self.diffusion_batch_mul, 1)

        outputs.loss = self.diffloss(z=z, target=target)
        outputs.diff_loss = outputs.loss
        return outputs

    def generate(self, batch, n_return_sequences=1):
        outputs = self.forward(batch, return_loss=False)
        condition = outputs.final_states
        sample_condition = condition
        if self.cfg_scale != 1.0:
            sample_condition = torch.cat(
                [
                    condition,
                    self.uncond_condition.to(dtype=condition.dtype)
                    .view(1, -1)
                    .expand_as(condition),
                ],
                dim=0,
            )
        generated_embs = self.diffloss.sample(
            sample_condition,
            temperature=self.diff_temperature,
            cfg=self.cfg_scale,
        )
        generated_embs = generated_embs[: condition.shape[0]]
        item_logits = self._item_similarity_logits(generated_embs)
        preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
        return preds.unsqueeze(-1)
