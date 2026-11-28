"""
Recent time-series architectures as benchmarks, reimplemented for this pipeline: a window of F input series
over W bars is mapped to one scalar, the next-bar target. The models are written from the descriptions in
the papers; no code is copied. Where the pipeline differs from the papers, the difference is stated.

  DLinear                Zeng et al. (2023), "Are Transformers effective for time series forecasting?".
                         Every input series is split into a moving-average trend and a remainder, and each
                         part has its own linear map. The paper forecasts every series from its own two
                         parts; our target is one scalar, so the parts of all series feed one output.
  PatchTSTRegressor      Nie et al. (2023), "A time series is worth 64 words". Every input series is cut
                         into overlapping patches, a patch is a token, and one Transformer encoder shared
                         by all series reads the tokens of a series (channel independence). The paper
                         forecasts a series from its own tokens; here the encoded tokens of all series are
                         flattened into one linear head.
  ITransformerRegressor  Liu et al. (2024), "iTransformer: inverted Transformers are effective for time
                         series forecasting". A whole input series is one token (a linear map of its W
                         values), and the encoder attends across the series. The paper projects a token to
                         the future of its own series; here the tokens of all series are flattened into one
                         linear head.

Both Transformers use pre-norm encoder layers with a feed-forward width of twice the token width, as the
Transformer of the paper does, and no normalisation of the window: all inputs are stationary and z-scored
once on the training block, and a per-window normalisation would remove the volatility level that is to
be forecast.
"""

import torch
import torch.nn as nn


class DLinear(nn.Module):
    def __init__(self, input_size: int, seq_len: int, kernel: int | None = None):
        super().__init__()
        self.kernel = kernel or max(3, (seq_len // 6) | 1)       # odd; a day of 4-hour bars for W = 42
        self.trend = nn.Linear(input_size * seq_len, 1)
        self.remainder = nn.Linear(input_size * seq_len, 1, bias=False)

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Moving average over time with the ends repeated, and what is left of the series."""
        pad = (self.kernel - 1) // 2
        padded = torch.cat([x[:, :1].expand(-1, pad, -1), x, x[:, -1:].expand(-1, pad, -1)], dim=1)
        trend = padded.unfold(1, self.kernel, 1).mean(-1)          # (batch, W, F)
        return trend, x - trend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend, rest = self.decompose(x)
        return self.trend(trend.flatten(1)) + self.remainder(rest.flatten(1))


def _encoder(d_model: int, nhead: int, num_layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model, dropout=dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)


class PatchTSTRegressor(nn.Module):
    def __init__(self, input_size: int, d_model: int, seq_len: int, patch_len: int = 8, stride: int = 4,
                 nhead: int = 2, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (seq_len + stride - patch_len) // stride + 1     # the end is padded by one stride
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.encoder = _encoder(d_model, nhead, num_layers, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(input_size * self.n_patches * d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, n_series = x.shape
        series = x.transpose(1, 2)                                                   # (batch, F, W)
        series = torch.cat([series, series[..., -1:].expand(-1, -1, self.stride)], dim=-1)
        patches = series.unfold(-1, self.patch_len, self.stride)                     # (batch, F, patches, patch_len)
        tokens = self.embed(patches).flatten(0, 1) + self.pos                        # every series on its own
        encoded = self.norm(self.encoder(tokens)).reshape(batch, n_series, -1)
        return self.head(encoded.flatten(1))


class ITransformerRegressor(nn.Module):
    def __init__(self, input_size: int, d_model: int, seq_len: int, nhead: int = 2, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)                  # one token per input series
        self.encoder = _encoder(d_model, nhead, num_layers, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(input_size * d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(x.transpose(1, 2))                    # (batch, F, d_model)
        return self.head(self.norm(self.encoder(tokens)).flatten(1))
