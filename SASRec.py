import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SASRecConfig:
    num_items: int
    max_seq_len: int = 25
    hidden_size: int = 128
    num_blocks: int = 3
    num_heads: int = 4
    dropout_rate: float = 0.2
    initializer_range: float = 0.02


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        return out.transpose(1, 2)


class SASRecBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout_rate)

        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = PointWiseFeedForward(hidden_size, dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.attn_layer_norm(x)
        attn_output, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = residual + self.attn_dropout(attn_output)

        residual = x
        x_norm = self.ffn_layer_norm(x)
        x = residual + self.ffn(x_norm)
        return x


class SASRec(nn.Module):
    def __init__(self, config: SASRecConfig):
        super().__init__()
        self.config = config
        self.num_items = config.num_items
        self.max_seq_len = config.max_seq_len
        self.hidden_size = config.hidden_size

        self.item_embedding = nn.Embedding(
            num_embeddings=self.num_items + 1,
            embedding_dim=self.hidden_size,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=self.max_seq_len,
            embedding_dim=self.hidden_size,
        )
        self.embedding_dropout = nn.Dropout(config.dropout_rate)

        self.blocks = nn.ModuleList([
            SASRecBlock(config.hidden_size, config.num_heads, config.dropout_rate)
            for _ in range(config.num_blocks)
        ])
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def log2feats(self, input_seq: torch.Tensor) -> torch.Tensor:
        device = input_seq.device
        batch_size, seq_len = input_seq.shape

        if seq_len > self.max_seq_len:
            input_seq = input_seq[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        x = self.item_embedding(input_seq) * math.sqrt(self.hidden_size)
        x = x + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        padding_mask = (input_seq == 0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, causal_mask=causal_mask)

        x = self.final_layer_norm(x)
        x = x * (~padding_mask).unsqueeze(-1)
        return x

    def forward(self, input_seq, positive_items, negative_items):
        seq_out = self.log2feats(input_seq)
        pos_emb = self.item_embedding(positive_items)
        neg_emb = self.item_embedding(negative_items)
        return seq_out, (seq_out * pos_emb).sum(-1), (seq_out * neg_emb).sum(-1)

    def calculate_loss(self, input_seq, positive_items, negative_items) -> torch.Tensor:
        _, pos_logits, neg_logits = self.forward(input_seq, positive_items, negative_items)
        valid_mask = (positive_items != 0).float()
        loss = (
            F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="none")
            + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction="none")
        ) * valid_mask
        return loss.sum() / torch.clamp(valid_mask.sum(), min=1.0)

    def get_last_hidden_state(self, input_seq: torch.Tensor) -> torch.Tensor:
        seq_out = self.log2feats(input_seq)
        non_pad_idx = torch.clamp((input_seq != 0).sum(dim=1) - 1, min=0)
        batch_idx = torch.arange(input_seq.size(0), device=input_seq.device)
        return seq_out[batch_idx, non_pad_idx]

    def predict(self, input_seq: torch.Tensor, candidate_items: Optional[torch.Tensor] = None) -> torch.Tensor:
        last_hidden = self.get_last_hidden_state(input_seq)
        if candidate_items is None:
            return torch.matmul(last_hidden, self.item_embedding.weight[1:].t())
        candidate_emb = self.item_embedding(candidate_items)
        return (last_hidden.unsqueeze(1) * candidate_emb).sum(-1)