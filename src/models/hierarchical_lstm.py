"""
Hierarchical real-valued LSTM (H-LSTM): the real-valued control for the
hierarchical quaternion LSTM.

Identical to HierarchicalQLSTM except that each semantic group is encoded by
a real nn.LSTM (4 inputs -> H units) instead of a QLSTM. The same fusion
modules are used, so encoder algebra (quaternion vs real) and fusion algebra
(Hamilton / learned PHM / dense / concat / group attention) can be varied
independently.

With group_projection=True each group's final state passes through a learned
H -> H linear map before fusion, mirroring the 4H -> H summary projection of
the quaternion encoders (control for that structural asymmetry).

With init="glorot" the recurrent weights are drawn as those of the quaternion layers
(src/models/quaternion_ops.QuaternionLinear): normal with variance 2 / (fan_in + fan_out)
of the real weight matrix, and zero biases, before the forget-gate bias is set. The
default keeps the uniform initialisation of nn.LSTM (control for that difference).
"""

import math

import torch
import torch.nn as nn

from .hierarchical_qlstm import _FUSION_REGISTRY


class HierarchicalLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_groups: int = 4, features_per_group: int = 4,
                 fusion_type: str = "concat", group_projection: bool = False, num_layers: int = 1,
                 init: str = "default"):
        super().__init__()
        self.num_groups, self.features_per_group = num_groups, features_per_group
        self.lstms = nn.ModuleList([
            nn.LSTM(features_per_group, hidden_size, num_layers=num_layers, batch_first=True)
            for _ in range(num_groups)
        ])
        if init not in ("default", "glorot"):
            raise ValueError(f"unknown init {init}")
        for lstm in self.lstms:  # forget-gate bias +1 in both bias vectors, as in RealLSTM
            with torch.no_grad():
                for name, p in lstm.named_parameters():
                    if init == "glorot":      # as QuaternionLinear.reset_parameters, for the real matrix
                        if "weight" in name:
                            p.normal_(0.0, math.sqrt(2.0 / (p.shape[0] + p.shape[1])))
                        else:
                            p.zero_()
                    if "bias" in name:
                        p[hidden_size:2 * hidden_size].fill_(1.0)
        self.projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_groups)
        ]) if group_projection else None
        self.fusion = _FUSION_REGISTRY[fusion_type](hidden_size, num_groups)
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        groups = x.view(b, s, self.num_groups, self.features_per_group)
        summaries = [self.lstms[g](groups[:, :, g])[0][:, -1] for g in range(self.num_groups)]
        if self.projections is not None:
            summaries = [proj(z) for proj, z in zip(self.projections, summaries)]
        return self.output_head(self.fusion(summaries))
