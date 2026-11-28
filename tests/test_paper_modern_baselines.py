"""Recent time-series benchmarks of the paper pipeline (src/models/modern_baselines.py)."""

import pytest
import torch

from src.models.modern_baselines import DLinear, ITransformerRegressor, PatchTSTRegressor
from src.paper.models import build_matched, n_params


@pytest.mark.parametrize("name", ["dlinear", "patchtst", "itransformer"])
def test_forward_shape_and_gradient(name):
    model, _ = build_matched(name, 3, 20000, seq_len=42)          # 12 features, as candidate A
    x = torch.randn(5, 42, 12)
    y = model(x)
    assert y.shape == (5, 1)
    y.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", ["patchtst", "itransformer"])
def test_width_meets_the_parameter_budget(name):
    model, _ = build_matched(name, 3, 20000, seq_len=42)
    assert abs(n_params(model) - 20000) / 20000 < 0.10


def test_dlinear_decomposition_adds_up_and_is_linear():
    model = DLinear(12, 42)
    x = torch.randn(4, 42, 12)
    trend, rest = model.decompose(x)
    torch.testing.assert_close(trend + rest, x)
    assert n_params(model) == 2 * 12 * 42 + 1
    a, b = torch.randn(4, 42, 12), torch.randn(4, 42, 12)
    bias = model(torch.zeros(1, 42, 12))
    torch.testing.assert_close(model(a + b) - bias, (model(a) - bias) + (model(b) - bias), atol=1e-5, rtol=1e-4)


def test_patchtst_reads_every_series_with_the_same_encoder():
    """Channel independence: before the head, a series is encoded without regard to the other series."""
    model = PatchTSTRegressor(3, 16, 42).eval()
    x = torch.randn(2, 42, 3)
    changed = x.clone()
    changed[..., 2] += 1.0                                        # only the third series changes

    def encoded(v):
        series = torch.cat([v.transpose(1, 2), v.transpose(1, 2)[..., -1:].expand(-1, -1, model.stride)], dim=-1)
        tokens = model.embed(series.unfold(-1, model.patch_len, model.stride)).flatten(0, 1) + model.pos
        return model.norm(model.encoder(tokens)).reshape(2, 3, -1)

    with torch.no_grad():
        a, b = encoded(x), encoded(changed)
    torch.testing.assert_close(a[:, :2], b[:, :2])
    assert not torch.allclose(a[:, 2], b[:, 2])


def test_itransformer_has_one_token_per_series():
    model = ITransformerRegressor(12, 16, 42)
    assert model.embed.in_features == 42 and model.head.in_features == 12 * 16
