"""
Quaternion Transformer encoders and their hierarchical variants.

Quaternion-valued self-attention in the spirit of Tay et al. (2019): every
linear map of a pre-norm Transformer encoder layer (Q, K, V, output and the
two feed-forward maps) is a quaternion linear layer, i.e. a real map with a
Hamilton-block weight matrix (QuaternionLinear.real_matrix), applied to
token vectors laid out as [quaternion unit][component]. Attention scores are
the Euclidean inner products of the flattened quaternion vectors (the real
part of the quaternion inner product), split into heads along quaternion
units. Layer normalisation acts on the flattened real vector.

Models
  QTransformerRegressor          all groups as input quaternions of one encoder
  HierarchicalTransformer        one encoder per semantic group (quaternion or
                                 real), then a fusion module from
                                 hierarchical_qlstm (concat, Hamilton meta, ...)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .quaternion_ops import QuaternionLinear
from .hierarchical_qlstm import _FUSION_REGISTRY


class QLinear(nn.Module):
    """Quaternion linear layer on flattened inputs (..., 4*in) -> (..., 4*out)."""

    def __init__(self, in_q: int, out_q: int):
        super().__init__()
        self.q = QuaternionLinear(in_q, out_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.q.real_matrix(), self.q.bias.reshape(-1))


class QEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with quaternion linear maps."""

    def __init__(self, d_q: int, nhead: int, ff_mult: int = 2):
        super().__init__()
        assert d_q % nhead == 0
        self.d_q, self.nhead = d_q, nhead
        self.qkv = QLinear(d_q, 3 * d_q)
        self.out = QLinear(d_q, d_q)
        self.ff1 = QLinear(d_q, ff_mult * d_q)
        self.ff2 = QLinear(ff_mult * d_q, d_q)
        self.norm1 = nn.LayerNorm(4 * d_q)
        self.norm2 = nn.LayerNorm(4 * d_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, S, 4*d_q)
        b, s, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, self.nhead, -1).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))       # (B, heads, S, 4*d_q/heads)
        att = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(att.transpose(1, 2).reshape(b, s, -1))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))


def _real_encoder(d_model: int, nhead: int, num_layers: int, activation: str = "relu") -> nn.Module:
    layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model, dropout=0.0,
                                       activation=activation, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)


class _GroupEncoder(nn.Module):
    """Encodes one sequence of `in_q` input quaternions; returns the last-token state (4*d_q)."""

    def __init__(self, in_q: int, d_q: int, seq_len: int, nhead: int, num_layers: int, quaternion: bool,
                 width: int | None = None, activation: str = "relu"):
        super().__init__()
        # real encoders may use a free width (matched-budget control); quaternion ones use 4 * d_q
        d = 4 * d_q if width is None else width
        self.embed = QLinear(in_q, d_q) if quaternion else nn.Linear(4 * in_q, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        self.layers = (nn.Sequential(*[QEncoderLayer(d_q, nhead) for _ in range(num_layers)])
                       if quaternion else _real_encoder(d, nhead, num_layers, activation))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, S, 4*in_q)
        h = self.embed(x) + self.pos[:, -x.shape[1]:]
        return self.norm(self.layers(h))[:, -1]


class QTransformerRegressor(nn.Module):
    """Flat quaternion Transformer: one encoder over all groups (early fusion)."""

    def __init__(self, n_groups: int, d_q: int, seq_len: int, nhead: int = 2, num_layers: int = 2):
        super().__init__()
        self.encoder = _GroupEncoder(n_groups, d_q, seq_len, nhead, num_layers, quaternion=True)
        self.head = nn.Linear(4 * d_q, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class HierarchicalTransformer(nn.Module):
    """One (quaternion or real) Transformer encoder per group, then fusion."""

    def __init__(self, n_groups: int, d_q: int, seq_len: int, fusion_type: str, quaternion: bool,
                 nhead: int = 2, num_layers: int = 1, width: int | None = None, activation: str = "relu"):
        super().__init__()
        self.n_groups = n_groups
        self.encoders = nn.ModuleList([
            _GroupEncoder(1, d_q, seq_len, nhead, num_layers, quaternion, width, activation)
            for _ in range(n_groups)
        ])
        enc_width = 4 * d_q if width is None else width
        self.projections = nn.ModuleList([nn.Linear(enc_width, d_q) for _ in range(n_groups)])
        self.fusion = _FUSION_REGISTRY[fusion_type](d_q, n_groups)
        self.head = nn.Linear(d_q, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = x.view(x.shape[0], x.shape[1], self.n_groups, 4)
        z = [p(enc(groups[:, :, g])) for g, (enc, p) in enumerate(zip(self.encoders, self.projections))]
        return self.head(self.fusion(z))
