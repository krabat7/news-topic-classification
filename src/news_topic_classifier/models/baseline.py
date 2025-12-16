from __future__ import annotations

import torch
from torch import nn


class MeanPoolClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
