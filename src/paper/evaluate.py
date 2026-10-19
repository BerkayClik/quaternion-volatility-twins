"""
Out-of-sample evaluation of a paper run directory.

Test blocks of all folds are disjoint and chronological, so each
(dataset, task, groupset, model, seed) gets one concatenated OOS series.

Metrics
  range : MSE / MAE on log range, QLIKE on the implied variance proxy
          (sigma^2 ~ exp(2 * log range)), R2_oos vs persistence and vs HAR
  return: MSE, R2_oos vs a zero forecast, directional accuracy (y != 0),
          Pesaran-Timmermann p-value
Tests
  Diebold-Mariano (Newey-West HAC variance) of each model's seed-averaged
  forecast against a benchmark, Holm-adjusted within (dataset, task).
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats


def _to_time(v: np.ndarray) -> pd.DatetimeIndex:
    """Epoch integers saved as ns (current runner) or us/s (older pandas defaults)."""
    big = np.abs(v).max() if len(v) else 0
    unit = "ns" if big > 1e17 else "us" if big > 1e14 else "s"
    return pd.to_datetime(v, unit=unit, utc=True)


def load_run(run_dir: str) -> pd.DataFrame:
    """Long frame: one row per (cell key, test timestamp)."""
    parts = []
    for path in glob.glob(os.path.join(run_dir, "preds", "*.npz")):
        ds, task, gs, model, seed, fold = os.path.basename(path)[:-4].split("__")
        z = np.load(path)
        parts.append(pd.DataFrame({
            "dataset": ds, "task": task, "groupset": gs, "model": model,
            "seed": int(seed[1:]), "fold": int(fold[1:]),
            "time": _to_time(z["test_time"]),
            "y": z["test_y"].astype(float), "pred": z["test_pred"].astype(float),
        }))
    return pd.concat(parts, ignore_index=True)


def qlike(log_range_true, log_range_pred):
    s2, h = np.exp(2 * log_range_true), np.exp(2 * log_range_pred)
    return s2 / h - np.log(s2 / h) - 1


def losses(task, y, p):
    return {"se": (p - y) ** 2, "ae": np.abs(p - y),
            **({"qlike": qlike(y, p)} if task == "range" else {})}


def pesaran_timmermann(y, p):
    """PT (1992) test of directional predictability; returns one-sided p-value."""
    m = y != 0
    y, p = np.sign(y[m]), np.sign(p[m])
    n = len(y)
    if n < 10 or np.all(p == p[0]):
        return np.nan
    py, pp = np.mean(y > 0), np.mean(p > 0)
    hit = np.mean(y == p)
    pstar = py * pp + (1 - py) * (1 - pp)
    v_hit = pstar * (1 - pstar) / n
    v_star = ((2 * py - 1) ** 2 * pp * (1 - pp) + (2 * pp - 1) ** 2 * py * (1 - py)) / n \
        + 4 * py * pp * (1 - py) * (1 - pp) / n ** 2
    den = v_hit - v_star
    if den <= 0:
        return np.nan
    return float(1 - stats.norm.cdf((hit - pstar) / np.sqrt(den)))


def diebold_mariano(loss_a, loss_b):
    """DM test of E[loss_a - loss_b] = 0 with Newey-West variance (Bartlett).
    Negative statistic => model a is more accurate. Two-sided p-value."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    lag = int(np.floor(n ** (1 / 3)))
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-seed metrics, then mean / std over seeds."""
    rows = []
    for key, g in df.groupby(["dataset", "task", "groupset", "model", "seed"]):
        g = g.sort_values("time")
        y, p = g.y.to_numpy(), g.pred.to_numpy()
        task = key[1]
        r = dict(zip(["dataset", "task", "groupset", "model", "seed"], key), n=len(g))
        L = losses(task, y, p)
        r.update(mse=L["se"].mean(), mae=L["ae"].mean())
        if task == "range":
            r["qlike"] = L["qlike"].mean()
        else:
            m = y != 0
            r["da"] = np.mean(np.sign(p[m]) == np.sign(y[m]))
            r["pt_p"] = pesaran_timmermann(y, p)
            r["r2_oos_zero"] = 1 - L["se"].sum() / np.sum(y ** 2)
        rows.append(r)
    per_seed = pd.DataFrame(rows)
    keys = ["dataset", "task", "groupset", "model"]
    agg = per_seed.groupby(keys).agg(["mean", "std"]).drop(columns=["seed"], level=0)
    agg.columns = [f"{a}_{b}" if b == "std" else a for a, b in agg.columns]
    return agg.reset_index()


def seed_mean(df):
    return (df.groupby(["dataset", "task", "groupset", "model", "time"], as_index=False)
              .agg(y=("y", "first"), pred=("pred", "mean")))


def dm_table(df: pd.DataFrame, benchmark: dict, loss: str = "se") -> pd.DataFrame:
    """DM of every model vs `benchmark[task]` = (groupset, model) on the same
    dataset/task and common timestamps; Holm-adjusted within dataset/task."""
    ens = seed_mean(df)
    out = []
    for (ds, task), g in ens.groupby(["dataset", "task"]):
        bgs, bmodel = benchmark[task]
        b = g[(g.groupset == bgs) & (g.model == bmodel)].set_index("time")
        if b.empty:
            continue
        rows = []
        for (gs, model), h in g.groupby(["groupset", "model"]):
            if (gs, model) == (bgs, bmodel):
                continue
            h = h.set_index("time")
            t = h.index.intersection(b.index)
            la = losses(task, h.loc[t, "y"].to_numpy(), h.loc[t, "pred"].to_numpy())[loss]
            lb = losses(task, b.loc[t, "y"].to_numpy(), b.loc[t, "pred"].to_numpy())[loss]
            stat, p = diebold_mariano(la, lb)
            rows.append(dict(dataset=ds, task=task, groupset=gs, model=model, n=len(t),
                             rel_loss=la.mean() / lb.mean(), dm_stat=stat, p=p))
        if rows:
            r = pd.DataFrame(rows)
            r["p_holm"] = holm(r.p)
            out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def dm_cross(df: pd.DataFrame, a: tuple, b: tuple, task: str = "range", loss: str = "se") -> dict:
    """Aggregate DM test of model a vs model b across assets.

    a, b = (groupset, model). The per-asset loss differentials are averaged over
    assets on each calendar timestamp (assets share calendar-aligned folds), and
    the DM test with Newey-West variance is applied to that cross-sectional
    average, which accounts for contemporaneous correlation across assets.
    """
    ens = seed_mean(df[df.task == task])
    diffs = []
    for ds, g in ens.groupby("dataset"):
        ga = g[(g.groupset == a[0]) & (g.model == a[1])].set_index("time")
        gb = g[(g.groupset == b[0]) & (g.model == b[1])].set_index("time")
        t = ga.index.intersection(gb.index)
        if len(t) == 0:
            continue
        la = losses(task, ga.loc[t, "y"].to_numpy(), ga.loc[t, "pred"].to_numpy())[loss]
        lb = losses(task, gb.loc[t, "y"].to_numpy(), gb.loc[t, "pred"].to_numpy())[loss]
        diffs.append(pd.Series(la - lb, index=t, name=ds))
    d = pd.concat(diffs, axis=1).mean(axis=1).sort_index()
    stat, p = diebold_mariano(d.to_numpy(), np.zeros(len(d)))
    return dict(n_assets=len(diffs), n_times=len(d), mean_diff=float(d.mean()), dm_stat=stat, p=p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    pd.set_option("display.width", 220, "display.max_rows", 500)
    df = load_run(args.run_dir)
    s = summarize(df)
    print(s.round(5).to_string(index=False))
    bench = {"range": ("P", "har"), "return": ("P", "persistence")}
    print("\nDM vs benchmark", bench)
    print(dm_table(df, bench).round(4).to_string(index=False))
