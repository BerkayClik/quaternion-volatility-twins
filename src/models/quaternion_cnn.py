"""
Temporal convolutional encoders, real and quaternion: the third encoder family of
the twin protocol, next to the recurrent and the attention encoders.

The block structure follows the QCNN of Lao, Zhang and Tang (2025, Expert Systems
with Applications 281, 127607), reimplemented here for one-dimensional (time)
inputs: 3-tap convolutions with ReLU, a max-pool after every second convolution,
then a fully connected layer on the flattened feature map. Their quaternion
convolution is the rotation form of Zhu et al. (2018), which is defined for pure
(three-channel) quaternions. Three of our four feature groups are full
quaternions, so the quaternion layer here is the Hamilton-product convolution of
Gaudet and Maurice (2018) and Parcollet et al. (2019): a real convolution whose
kernel is built from Hamilton blocks, exactly as QuaternionLinear.real_matrix
does for linear layers. ReLU and max-pooling act on each component separately
(split activation), as in those papers.

The real and the quaternion network share every other choice: kernel size,
padding, pooling, depth, Glorot-normal initialisation and the linear read-out.

Models
  CNNRegressor       all groups as input channels of one encoder (early fusion)
  HierarchicalCNN    one encoder per semantic group, then a fusion module from
                     hierarchical_qlstm (concat, Hamilton meta, ...)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_qlstm import _FUSION_REGISTRY
from .quaternion_transformer import QLinear

KERNEL = 3


class QuaternionConv1d(nn.Module):
    """
    Hamilton-product convolution over time with 'same' zero padding.

    Input (batch, 4 * in_q, T) and output (batch, 4 * out_q, T); the channel index
    is quaternion * 4 + component, the layout of QuaternionLinear.real_matrix.
    Each kernel tap holds one weight quaternion w per (output, input) pair and the
    layer sums the products x (x) w over inputs and taps (input on the left, the
    convention of QuaternionLinear), so it has a quarter of the weights of a real
    convolution of the same width.
    """

    def __init__(self, in_q: int, out_q: int, kernel_size: int = KERNEL):
        super().__init__()
        self.in_q, self.out_q, self.kernel_size = in_q, out_q, kernel_size
        self.weight = nn.Parameter(torch.empty(out_q, in_q, kernel_size, 4))
        self.bias = nn.Parameter(torch.zeros(out_q, 4))
        # Glorot normal for the equivalent real kernel of shape (4 * out_q, 4 * in_q, k)
        nn.init.normal_(self.weight, 0.0, math.sqrt(2.0 / (4 * (in_q + out_q) * kernel_size)))

    def real_kernel(self) -> torch.Tensor:
        """The same map as a real Conv1d kernel of shape (4 * out_q, 4 * in_q, k)."""
        w_r, w_i, w_j, w_k = self.weight.unbind(-1)                      # each (out, in, k)
        rows = [torch.cat(blocks, dim=1) for blocks in (
            [w_r, -w_i, -w_j, -w_k], [w_i, w_r, w_k, -w_j],
            [w_j, -w_k, w_r, w_i], [w_k, w_j, -w_i, w_r])]               # one per output component
        # (comp_out, out, comp_in * in, k) -> (out, comp_out, in, comp_in, k)
        m = torch.stack(rows).view(4, self.out_q, 4, self.in_q, self.kernel_size).permute(1, 0, 3, 2, 4)
        return m.reshape(4 * self.out_q, 4 * self.in_q, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.real_kernel(), self.bias.reshape(-1), padding=self.kernel_size // 2)


def _real_conv(in_channels: int, out_channels: int) -> nn.Conv1d:
    """Real twin of QuaternionConv1d: same kernel, padding and initialisation rule."""
    conv = nn.Conv1d(in_channels, out_channels, KERNEL, padding=KERNEL // 2)
    nn.init.xavier_normal_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


def _real_linear(in_features: int, out_features: int) -> nn.Linear:
    """Real twin of QLinear (Glorot normal weights, zero bias)."""
    lin = nn.Linear(in_features, out_features)
    nn.init.xavier_normal_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class ConvEncoder(nn.Module):
    """
    `depth` convolutions with ReLU, a max-pool(2) after every second one, then a
    fully connected layer with ReLU on the flattened feature map.

    `channels` counts quaternion channels for the quaternion encoder and real
    channels for the real one; the encoder returns `out_dim` real features
    (4 * channels for the quaternion encoder, channels for the real one).
    """

    def __init__(self, in_q: int, channels: int, seq_len: int, depth: int = 4, quaternion: bool = False):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers, length = [], seq_len
        for layer in range(depth):
            if quaternion:
                layers.append(QuaternionConv1d(in_q if layer == 0 else channels, channels))
            else:
                layers.append(_real_conv(4 * in_q if layer == 0 else channels, channels))
            layers.append(nn.ReLU())
            if layer % 2 == 1 and length >= 2:
                layers.append(nn.MaxPool1d(2))
                length //= 2
        self.convs = nn.Sequential(*layers)
        self.fc = QLinear(length * channels, channels) if quaternion else _real_linear(length * channels, channels)
        self.out_dim = 4 * channels if quaternion else channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (batch, T, 4 * in_q)
        maps = self.convs(x.transpose(1, 2))                      # (batch, C, T')
        # time-major flattening keeps the four components of a quaternion adjacent for QLinear
        return F.relu(self.fc(maps.transpose(1, 2).flatten(1)))


class CNNRegressor(nn.Module):
    """Flat (quaternion or real) temporal CNN: one encoder over all groups."""

    def __init__(self, n_groups: int, channels: int, seq_len: int, depth: int = 4, quaternion: bool = False):
        super().__init__()
        self.encoder = ConvEncoder(n_groups, channels, seq_len, depth, quaternion)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class HierarchicalCNN(nn.Module):
    """
    One (quaternion or real) temporal CNN per group, then fusion.

    As in the recurrent pair (HierarchicalQLSTM / HierarchicalLSTM), the quaternion
    encoders pass through a real 4H -> H projection before fusion and the real
    encoders, whose output already has H units, do not.
    """

    def __init__(self, n_groups: int, channels: int, seq_len: int, fusion_type: str, quaternion: bool,
                 depth: int = 4):
        super().__init__()
        self.n_groups = n_groups
        self.encoders = nn.ModuleList([
            ConvEncoder(1, channels, seq_len, depth, quaternion) for _ in range(n_groups)
        ])
        self.projections = nn.ModuleList([
            nn.Linear(4 * channels, channels) for _ in range(n_groups)
        ]) if quaternion else None
        self.fusion = _FUSION_REGISTRY[fusion_type](channels, n_groups)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = x.view(x.shape[0], x.shape[1], self.n_groups, 4)
        z = [enc(groups[:, :, g]) for g, enc in enumerate(self.encoders)]
        if self.projections is not None:
            z = [proj(v) for proj, v in zip(self.projections, z)]
        return self.head(self.fusion(z))
