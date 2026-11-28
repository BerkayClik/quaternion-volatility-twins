"""Cascade support of the pooled training routine: the network is trained on the residual of an earlier stage
(fit_predict_pooled(..., offsets=...)) and its forecast is the stage-1 forecast plus the network."""

import copy

import numpy as np
import pytest
import torch

from src.paper.train import fit_predict_pooled

WINDOW, CFG = 8, {"lr": 0.01, "batch_size": 32, "max_epochs": 4, "patience": 2, "grad_clip": 1.0}


class Flat(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.out = torch.nn.Linear(n_features * WINDOW, 1)

    def forward(self, x):
        return self.out(x.flatten(1))


def _panel(n_assets=2, n=300, f=3):
    rng = np.random.default_rng(0)
    Xs = [rng.normal(size=(n, f)).astype(np.float32) for _ in range(n_assets)]
    ys = [(X[:, 0] * 0.5 + rng.normal(scale=0.5, size=n)).astype(np.float32) for X in Xs]
    fold = {"train": np.arange(0, 200), "val": np.arange(201, 250), "test": np.arange(251, 299)}
    return Xs, ys, [fold] * n_assets


def _fit(Xs, ys, folds, offsets=None):
    torch.manual_seed(0)
    return fit_predict_pooled(Flat(Xs[0].shape[1]), Xs, ys, folds, WINDOW, CFG, "cpu", 0, offsets=offsets)


def test_zero_offset_changes_nothing():
    Xs, ys, folds = _panel()
    plain = _fit(Xs, ys, folds)
    zero = _fit(Xs, ys, folds, offsets=[np.zeros(len(y), np.float32) for y in ys])
    for a, b in zip(plain, zero):
        np.testing.assert_allclose(a["test_pred"], b["test_pred"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(a["val_pred"], b["val_pred"], rtol=0, atol=1e-6)


def test_cascade_is_stage_one_plus_the_network_trained_on_the_residual():
    Xs, ys, folds = _panel()
    rng = np.random.default_rng(1)
    offsets = [(0.8 * y + rng.normal(scale=0.1, size=len(y))).astype(np.float32) for y in ys]
    cascade = _fit(Xs, ys, folds, offsets=copy.deepcopy(offsets))
    residual = _fit(Xs, [y - o for y, o in zip(ys, offsets)], folds)
    for c, r, o in zip(cascade, residual, offsets):
        np.testing.assert_array_equal(c["test_rows"], r["test_rows"])
        np.testing.assert_allclose(c["test_pred"], r["test_pred"] + o[r["test_rows"]], rtol=0, atol=1e-6)
        np.testing.assert_allclose(c["val_pred"], r["val_pred"] + o[r["val_rows"]], rtol=0, atol=1e-6)


def test_rows_outside_the_blocks_may_lack_a_stage_one_forecast_but_used_rows_may_not():
    Xs, ys, folds = _panel()
    offsets = [np.zeros(len(y), np.float32) for y in ys]
    offsets[0][[200, 250, 299]] = np.nan                     # purged rows and the last row: never used
    plain = _fit(Xs, ys, folds)
    gaps = _fit(Xs, ys, folds, offsets=offsets)
    np.testing.assert_allclose(plain[0]["test_pred"], gaps[0]["test_pred"], rtol=0, atol=1e-6)
    offsets[1][:WINDOW - 1] = np.nan                         # first bars of an asset: no window ends there
    early = _fit(Xs, ys, folds, offsets=offsets)             # (the residual there is zero: it shifts the scaling constants only)
    assert all(np.isfinite(r["test_pred"]).all() and np.isfinite(r["val_pred"]).all() for r in early)
    offsets[1][100] = np.nan                                 # a training row where a window ends
    with pytest.raises(ValueError):
        _fit(Xs, ys, folds, offsets=offsets)
