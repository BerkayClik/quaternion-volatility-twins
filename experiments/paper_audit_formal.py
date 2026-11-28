"""
Formal version of the LunarCrush look-ahead audit (Appendix A.2), answering two
points of the second Deep Review:

1. Out-of-sample residuals. The HAR volatility innovation and the AR pre-whitening
   residual of every feature use coefficients estimated only on data before the bar
   (expanding window, refitted every REFIT_DAYS, first BURN_DAYS of each period as
   burn-in), instead of one in-sample regression over the whole series.
2. A formal test of the ordering. For every pre-whitened feature u and target z
   (return or HAR innovation), D = |corr(u_t, z_{t+1})| - |corr(u_t, z_t)|. A timing
   error (u_t containing bar t+1 information) makes D > 0. H0: D <= 0 is tested with a
   moving-block bootstrap of the rows (u_t, z_t, z_{t+1}), which keeps serial
   dependence and heteroskedasticity; one-sided p = share of bootstrap D* <= 0.
   Holm adjustment over the eight features of a series (coin, frequency, period).
   The one-bar-forward Galaxy Score placebo measures the power of the test.

    .venv/bin/python -W ignore experiments/paper_audit_formal.py
Output: docs/paper/manuscript/tables/r8_audit_formal.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.paper.baselines import HAR_LAGS  # noqa: E402
from src.paper.data import FFILL_LIMIT, GROUPS, build_features, load_raw  # noqa: E402

COINS = ["btc", "eth", "sol", "bnb", "xrp", "ada", "doge", "link", "ltc", "dot", "avax", "trx", "bch", "xlm", "atom"]
PERIODS = {"2020-2022": ("2020-01-01", "2022-12-31"), "2023-2026": ("2023-01-01", "2026-10-14")}
AR_ORDER = {"1d": 7, "4h": 6, "1h": 24}       # as in the in-sample audit
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24}
BLOCK = {"1d": 10, "4h": 30, "1h": 72}        # bootstrap block length in bars (~ n^(1/3) or more)
REFIT_DAYS, BURN_DAYS, REPS, ALPHA = 30, 180, 999, 0.05
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "paper", "manuscript", "tables", "r8_audit_formal.csv")


def recursive_ols_resid(y: np.ndarray, X: np.ndarray, refit: int, burn: int) -> np.ndarray:
    """Residual y_t - x_t'b, with b estimated by OLS on rows < start of t's refit block
    (expanding window). Rows with missing values are skipped in the fits; the first
    `burn` rows get NaN."""
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    ok = np.isfinite(y) & np.isfinite(A).all(axis=1)
    res = np.full(n, np.nan)
    for start in range(burn, n, refit):
        fit = ok.copy()
        fit[start:] = False
        if fit.sum() < 10 * A.shape[1]:
            continue
        b = np.linalg.lstsq(A[fit], y[fit], rcond=None)[0]
        blk = slice(start, min(start + refit, n))
        res[blk] = np.where(ok[blk], y[blk] - A[blk] @ b, np.nan)
    return res


def har_innovation(log_range: np.ndarray, lags, refit, burn) -> np.ndarray:
    """Element t: innovation of bar t+1 given information up to bar t (recursive HAR)."""
    lr = pd.Series(log_range)
    X = pd.concat([lr.rolling(k).mean() for k in lags], axis=1).to_numpy()
    return recursive_ols_resid(lr.shift(-1).to_numpy(), X, refit, burn)


def prewhiten(x: np.ndarray, p, refit, burn) -> np.ndarray:
    """Element t: part of x_t not predictable from x_{t-1..t-p} (recursive AR(p))."""
    s = pd.Series(x)
    X = pd.concat([s.shift(k) for k in range(1, p + 1)], axis=1).to_numpy()
    return recursive_ols_resid(x, X, refit, burn)


def _corr(u, z):
    """Column-wise Pearson correlation along axis -1 (bootstrap draws in leading axes)."""
    u = u - u.mean(-1, keepdims=True)
    z = z - z.mean(-1, keepdims=True)
    return (u * z).sum(-1) / np.sqrt((u * u).sum(-1) * (z * z).sum(-1))


def block_indices(n, block, reps, rng):
    starts = rng.integers(0, n - block + 1, size=(reps, -(-n // block)))
    return (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]


def ordering_test(u, z_same, z_next, block, rng):
    """D = |corr(u, z_next)| - |corr(u, z_same)| and its one-sided bootstrap p-value (H0: D <= 0)."""
    d = abs(_corr(u, z_next)) - abs(_corr(u, z_same))
    ds = []
    for reps in np.diff(np.r_[0:REPS:100, REPS]):    # chunks of draws keep memory small
        idx = block_indices(len(u), block, reps, rng)
        ds.append(np.abs(_corr(u[idx], z_next[idx])) - np.abs(_corr(u[idx], z_same[idx])))
    ds = np.concatenate(ds)
    return float(d), float((1 + np.sum(ds <= 0)) / (REPS + 1))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def main():
    feats = GROUPS["social"] + GROUPS["sentiment"]
    rng = np.random.default_rng(0)
    rows = []
    for f in ("1d", "4h", "1h"):
        refit, burn = REFIT_DAYS * BARS_PER_DAY[f], BURN_DAYS * BARS_PER_DAY[f]
        for coin in COINS:
            try:
                binance, lunar = load_raw(coin, f)
            except FileNotFoundError:
                continue
            fr = build_features(binance, lunar, FFILL_LIMIT[f]).replace([np.inf, -np.inf], np.nan)
            innov = har_innovation(fr["log_range"].to_numpy(float), HAR_LAGS[f], refit, burn)
            fr["innov_next"], fr["innov_same"] = innov, np.r_[np.nan, innov[:-1]]
            fr["placebo"] = fr["galaxy_score"].shift(-1)              # one-bar-forward leak
            for per, (s, e) in PERIODS.items():
                g = fr[(fr.index >= pd.Timestamp(s, tz=fr.index.tz)) &
                       (fr.index <= pd.Timestamp(e, tz=fr.index.tz) + pd.Timedelta("1D"))].copy()
                for col in feats + ["placebo"]:
                    if g[col].notna().sum() < burn + 200 or g[col].std() == 0:
                        continue
                    g["u"] = prewhiten(g[col].to_numpy(float), AR_ORDER[f], refit, burn)
                    x = g[["u", "ret", "y_return", "innov_same", "innov_next"]].dropna()
                    if len(x) < 200:
                        continue
                    u = x["u"].to_numpy()
                    row = dict(freq=f, coin=coin, period=per, feature=col, n=len(x))
                    for tgt, same, nxt in (("ret", "ret", "y_return"), ("innov", "innov_same", "innov_next")):
                        row[f"{tgt}_same"] = _corr(u, x[same].to_numpy())
                        row[f"{tgt}_next"] = _corr(u, x[nxt].to_numpy())
                        row[f"{tgt}_d"], row[f"{tgt}_p"] = ordering_test(
                            u, x[same].to_numpy(), x[nxt].to_numpy(), BLOCK[f], rng)
                    rows.append(row)
        print(f, flush=True)
    t = pd.DataFrame(rows)
    real = t.feature != "placebo"
    for tgt in ("ret", "innov"):  # Holm over the eight real features of each series
        t[f"{tgt}_p_holm"] = np.nan
        for _, g in t[real].groupby(["freq", "coin", "period"]):
            t.loc[g.index, f"{tgt}_p_holm"] = holm(g[f"{tgt}_p"])
        t.loc[~real, f"{tgt}_p_holm"] = t.loc[~real, f"{tgt}_p"]
    t.to_csv(OUT, index=False)

    for (f, per, is_real), g in t.groupby(["freq", "period", real]):
        rej = (g.ret_p_holm < ALPHA) | (g.innov_p_holm < ALPHA)
        tag = "features" if is_real else "PLACEBO "
        print(f"{tag} {f} {per}: n_series={g.coin.nunique()} rows={len(g)} "
              f"rejected ret {int((g.ret_p_holm < ALPHA).sum())} innov {int((g.innov_p_holm < ALPHA).sum())} "
              f"either {int(rej.sum())}/{len(g)}  min p ret {g.ret_p.min():.3f} innov {g.innov_p.min():.3f}  "
              f"max|ret_next| {g.ret_next.abs().max():.3f} max|innov_next| {g.innov_next.abs().max():.3f}")


if __name__ == "__main__":
    main()
