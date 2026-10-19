"""
Training / prediction for one (dataset, fold, model, seed) cell.

Protocol shared by every neural model:
  * inputs z-scored with train-fold statistics (constant columns left at 0)
  * target z-scored with train-fold statistics, MSE loss, predictions mapped back
  * Adam, same lr / batch / epochs for all families, shuffled training windows
  * early stopping on val loss, best-val weights restored (kept in memory)
"""

import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Windows(TorchDataset):
    """Sample for row t = X[t-W+1 .. t] -> y[t] (y already holds bar t+1)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, rows: np.ndarray, window: int):
        self.X, self.y, self.window = torch.from_numpy(X), torch.from_numpy(y), window
        self.rows = rows[rows >= window - 1]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t = self.rows[i]
        return self.X[t - self.window + 1: t + 1], self.y[t]


def standardize(X: np.ndarray, y: np.ndarray, train_rows: np.ndarray):
    mu, sd = X[train_rows].mean(0), X[train_rows].std(0)
    sd[sd < 1e-8] = 1.0
    ymu, ysd = y[train_rows].mean(), y[train_rows].std()
    Xs = ((X - mu) / sd).astype(np.float32)
    ys = ((y - ymu) / ysd).astype(np.float32)
    return Xs, ys, (float(ymu), float(ysd))


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = [model(xb.to(device)).squeeze(-1).cpu() for xb, _ in loader]
    return torch.cat(out).numpy()


def _train(model, Xs, ys, train_rows, val_rows, window, cfg, device, seed):
    """Fit on train windows (shuffled), early-stop on val MSE (standardized units)."""
    mk = lambda rows, shuffle: DataLoader(Windows(Xs, ys, rows, window), batch_size=cfg["batch_size"],
                                          shuffle=shuffle, generator=torch.Generator().manual_seed(seed))
    train_dl, val_dl = mk(train_rows, True), mk(val_rows, False)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    loss_fn = torch.nn.MSELoss()
    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tr = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
            tr += loss.item() * len(xb)
        val_pred = predict(model, val_dl, device)
        val_loss = float(np.mean((val_pred - val_dl.dataset.y[val_dl.dataset.rows].numpy()) ** 2))
        history.append((tr / len(train_dl.dataset), val_loss))
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"non-finite val loss at epoch {epoch}")
        if val_loss < best - 1e-6:
            best, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    info = {"best_epoch": int(np.argmin([h[1] for h in history])), "epochs": len(history),
            "best_val_mse_std": best}
    return model, info


def _predict_rows(model, Xs, ys, rows, window, cfg, device):
    ds = Windows(Xs, ys, rows, window)
    return ds.rows, predict(model, DataLoader(ds, batch_size=cfg["batch_size"]), device)


def fit_predict(model, X, y, fold, window, cfg, device, seed):
    """Train on fold['train'], early-stop on fold['val']; return test/val preds (original scale)."""
    set_seed(seed)
    Xs, ys, (ymu, ysd) = standardize(X, y, fold["train"])
    model, info = _train(model, Xs, ys, fold["train"], fold["val"], window, cfg, device, seed)
    te_rows, te = _predict_rows(model, Xs, ys, fold["test"], window, cfg, device)
    va_rows, va = _predict_rows(model, Xs, ys, fold["val"], window, cfg, device)
    return {"test_rows": te_rows, "test_pred": te * ysd + ymu,
            "val_rows": va_rows, "val_pred": va * ysd + ymu, **info}


def fit_predict_pooled(model, Xs_list, ys_list, folds, window, cfg, device, seed):
    """One model trained on the windows of several assets.

    Each asset is standardized with its own train-fold statistics, the arrays
    are concatenated, and windows never cross asset boundaries. Returns one
    result dict per asset, in the original target scale of that asset.
    """
    set_seed(seed)
    parts, stats, offsets, off = [], [], [], 0
    for X, y, fo in zip(Xs_list, ys_list, folds):
        Xs, ys, st = standardize(X, y, fo["train"])
        parts.append((Xs, ys)); stats.append(st); offsets.append(off); off += len(X)
    X_all = np.concatenate([p[0] for p in parts])
    y_all = np.concatenate([p[1] for p in parts])
    # shift rows by the asset offset and drop windows that would start in the previous asset
    shift = lambda key: np.concatenate([fo[key][fo[key] >= window - 1] + o for fo, o in zip(folds, offsets)])
    model, info = _train(model, X_all, y_all, shift("train"), shift("val"), window, cfg, device, seed)
    out = []
    for fo, o, (ymu, ysd) in zip(folds, offsets, stats):
        te_rows, te = _predict_rows(model, X_all, y_all, fo["test"][fo["test"] >= window - 1] + o, window, cfg, device)
        va_rows, va = _predict_rows(model, X_all, y_all, fo["val"][fo["val"] >= window - 1] + o, window, cfg, device)
        out.append({"test_rows": te_rows - o, "test_pred": te * ysd + ymu,
                    "val_rows": va_rows - o, "val_pred": va * ysd + ymu, **info})
    return out
