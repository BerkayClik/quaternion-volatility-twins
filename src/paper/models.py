"""
Model zoo for the paper, built on the validated modules in src/models.

All networks receive globally z-scored, stationary inputs, so normalization
layers are disabled (norm_type="none") and outputs are never denormalized
(target_mode != "price"). Every family is sized to a shared parameter budget.

Names:
  lstm, lstm_attn                 real-valued LSTM on all features
  qlstm, qlstm_attn               flat quaternion LSTM, one input quaternion per group
  hq_<fusion>[_attn]              hierarchical QLSTM: one QLSTM per group, then fusion
                                  fusion in {concat, gattn, meta}; meta needs 4 groups
  transformer                     2-layer pre-norm Transformer encoder, d_model = 4 * hidden
  qtransformer                    flat quaternion Transformer (2 layers, d_q = 2 * hidden quaternions)
  hqt_<fusion> / ht_<fusion>      hierarchical quaternion / real Transformer (1 layer, 1 head per group)
  hl_<fusion>                     hierarchical *real* LSTM control (one nn.LSTM per group)
  hlp_<fusion>                    H-LSTM with a learned H -> H projection per group before fusion
                                  (matches the 4H -> H summary projection of hq_<fusion>)
  transformer_g                   matched-twin control: flat real Transformer with GELU and 2 heads (as qtransformer), d_model = 2 * hidden
  htg_<fusion>                    matched-twin control: hierarchical real Transformer with GELU and a free
                                  encoder width w (hidden = w, summary = ceil(w / 4)), so the budget is met closely
  fusion phm / dense              learned-algebra (PHM) and unstructured real fusion controls
"""

import torch.nn as nn

from src.models.real_lstm import RealLSTM
from src.models.real_lstm_attention import RealLSTMAttention
from src.models.qnn_attention_model import QNNAttentionModel, QuaternionLSTMNoAttention
from src.models.hierarchical_qlstm import HierarchicalQLSTM
from src.models.hierarchical_lstm import HierarchicalLSTM
from src.models.transformer_encoder import TransformerRegressor
from src.models.quaternion_transformer import QTransformerRegressor, HierarchicalTransformer

FUSION = {"concat": "concat", "gattn": "group_attention", "meta": "meta_quaternion",
          "phm": "phm", "dense": "dense",
          # tanh after the fusion's mixing layer (nonlinear fusion)
          "metan": "meta_quaternion_act", "phmn": "phm_act", "densen": "dense_act"}
COMMON = dict(norm_type="none", target_mode="returns", target_col=0)


def build(name: str, n_groups: int, hidden: int, num_layers: int = 1, dropout: float = 0.0,
          seq_len: int = 168) -> nn.Module:
    n_feat = 4 * n_groups
    if name == "transformer":  # d_model must be divisible by the 4 heads
        return TransformerRegressor(n_feat, 4 * hidden, seq_len)
    if name == "transformer_g":
        return TransformerRegressor(n_feat, 2 * hidden, seq_len, nhead=2, activation="gelu")  # finer width steps
    if name.startswith("htg_"):
        fusion = FUSION[name[4:]]
        if fusion in ("meta_quaternion", "phm") and n_groups != 4:
            raise ValueError("4-component fusion needs exactly 4 groups")
        return HierarchicalTransformer(n_groups, -(-hidden // 4), seq_len, fusion, False, nhead=1,
                                       width=hidden, activation="gelu")
    if name == "lstm":
        return RealLSTM(n_feat, hidden, num_layers, dropout)
    if name == "lstm_attn":
        return RealLSTMAttention(n_feat, hidden, num_layers, dropout)
    if name in ("qlstm", "qlstm_attn"):
        cls = QNNAttentionModel if name == "qlstm_attn" else QuaternionLSTMNoAttention
        return cls(hidden_size=hidden, num_layers=num_layers, dropout=dropout,
                   input_size=n_groups, num_features=n_feat, **COMMON)
    if name == "qtransformer":
        return QTransformerRegressor(n_groups, 2 * hidden, seq_len)
    if name.startswith(("hqt_", "ht_")):
        quaternion = name.startswith("hqt_")
        fusion = FUSION[name.split("_", 1)[1]]
        if fusion in ("meta_quaternion", "phm") and n_groups != 4:
            raise ValueError("4-component fusion needs exactly 4 groups")
        return HierarchicalTransformer(n_groups, hidden, seq_len, fusion, quaternion, nhead=1)
    if name.startswith(("hl_", "hlp_")):
        fusion = FUSION[name.split("_", 1)[1]]
        if fusion in ("meta_quaternion", "phm") and n_groups != 4:
            raise ValueError("4-component fusion needs exactly 4 groups")
        return HierarchicalLSTM(hidden, num_groups=n_groups, fusion_type=fusion,
                                group_projection=name.startswith("hlp_"))
    if name.startswith("hq_"):
        parts = name[3:].split("_")
        fusion, attn = FUSION[parts[0]], parts[-1] == "attn"
        if fusion in ("meta_quaternion", "phm") and n_groups != 4:
            raise ValueError("4-component fusion needs exactly 4 groups")
        return HierarchicalQLSTM(hidden_size=hidden, num_layers=num_layers, dropout=dropout,
                                 num_features=n_feat, num_groups=n_groups, fusion_type=fusion,
                                 use_temporal_attention=attn, **COMMON)
    raise ValueError(f"unknown model {name}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_matched(name: str, n_groups: int, budget: int, **kw) -> tuple[nn.Module, int]:
    """Build `name` with the hidden size whose parameter count is closest to `budget`."""
    count = lambda h: n_params(build(name, n_groups, h, **kw))
    lo, hi = 2, 512                       # parameter count is monotone in hidden size
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if count(mid) <= budget else (lo, mid)
    best = min((lo, hi), key=lambda h: abs(count(h) - budget))
    return build(name, n_groups, best, **kw), best
