"""
Transformer-encoder baseline for windowed regression.

A standard pre-norm encoder (Vaswani et al., 2017) on the input window:
linear embedding + learned positional embedding, `num_layers` encoder
layers with `nhead` heads and a 2x feed-forward, then a linear head on the
last time step (causality is irrelevant here: the whole window is past data).
"""

import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):
    def __init__(self, input_size: int, d_model: int, seq_len: int, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model, dropout=dropout,
                                           activation=activation, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.pos[:, -x.shape[1]:]
        h = self.encoder(h)
        return self.head(self.norm(h[:, -1]))
