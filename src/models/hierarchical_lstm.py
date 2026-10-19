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
"""

import torch
import torch.nn as nn

from .hierarchical_qlstm import _FUSION_REGISTRY


class HierarchicalLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_groups: int = 4, features_per_group: int = 4,
                 fusion_type: str = "concat", group_projection: bool = False):
        super().__init__()
        self.num_groups, self.features_per_group = num_groups, features_per_group
        self.lstms = nn.ModuleList([
            nn.LSTM(features_per_group, hidden_size, batch_first=True) for _ in range(num_groups)
        ])
        for lstm in self.lstms:  # forget-gate bias +1 in both bias vectors, as in RealLSTM
            with torch.no_grad():
                for name, p in lstm.named_parameters():
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
