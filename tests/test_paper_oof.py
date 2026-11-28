"""Out-of-fold forecasts of training rows (data.time_blocks, data.held_out_fold; runner key `oof`): one block of
time of the training rows is forecast by a model trained on the other training rows."""

import numpy as np
import pandas as pd
import torch

from src.paper.data import held_out_fold, time_blocks
from src.paper.train import fit_predict_pooled

WINDOW, CFG, BLOCKS = 8, {"lr": 0.01, "batch_size": 32, "max_epochs": 3, "patience": 2, "grad_clip": 1.0}, 5


class Flat(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.out = torch.nn.Linear(n_features * WINDOW, 1)

    def forward(self, x):
        return self.out(x.flatten(1))


def _panel():
    """Two assets on one calendar; the second starts later, so it has no bar in the first block of time."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
    starts = [0, 150]
    indexes = [time[s:] for s in starts]
    Xs = [rng.normal(size=(len(i), 3)).astype(np.float32) for i in indexes]
    ys = [(X[:, 0] * 0.5 + rng.normal(scale=0.5, size=len(X))).astype(np.float32) for X in Xs]
    folds = [{"fold": 0, "train": np.arange(0, 300 - s), "val": np.arange(301 - s, 350 - s), "test": np.arange(351 - s, 399 - s)}
             for s in starts]
    return indexes, Xs, ys, folds


def test_the_blocks_partition_the_training_rows():
    indexes, _, _, folds = _panel()
    edges = time_blocks(indexes, folds, BLOCKS)
    for index, fold in zip(indexes, folds):
        held = [held_out_fold(index, fold, edges, j) for j in range(BLOCKS)]
        np.testing.assert_array_equal(np.sort(np.concatenate([h["test"] for h in held])), fold["train"])
        for h in held:
            np.testing.assert_array_equal(np.sort(np.concatenate([h["train"], h["test"]])), fold["train"])
            np.testing.assert_array_equal(h["val"], fold["val"])
            np.testing.assert_array_equal(h["scale"], fold["train"])
    assert len(held_out_fold(indexes[1], folds[1], edges, 0)["test"]) == 0      # the asset that starts later


def test_forecasts_are_those_of_the_held_out_rows_in_the_scaling_of_the_whole_training_block():
    indexes, Xs, ys, folds = _panel()
    edges = time_blocks(indexes, folds, BLOCKS)
    for block in (0, 2):
        held = [held_out_fold(i, fo, edges, block) for i, fo in zip(indexes, folds)]
        torch.manual_seed(0)
        out = fit_predict_pooled(Flat(3), Xs, ys, held, WINDOW, CFG, "cpu", 0)
        for res, h in zip(out, held):
            np.testing.assert_array_equal(res["test_rows"], h["test"][h["test"] >= WINDOW - 1])
            assert len(res["test_pred"]) == len(res["test_rows"]) and np.isfinite(res["test_pred"]).all()
        zero = Flat(3)
        for p in zero.parameters():
            torch.nn.init.zeros_(p)
        frozen = dict(CFG, lr=0.0, max_epochs=1)
        res = fit_predict_pooled(zero, Xs, ys, held, WINDOW, frozen, "cpu", 0)
        for r, y, fo in zip(res, ys, folds):                   # a network that returns zero forecasts the mean of the whole block
            if len(r["test_pred"]):
                np.testing.assert_allclose(r["test_pred"], y[fo["train"]].mean(), rtol=0, atol=1e-5)
