"""
Economic evaluation of saved forecasts (no retraining).

Volatility forecasts (range task)
  * The predicted next-bar log range L is mapped to a Parkinson (1980)
    variance, sigma^2 = exp(L)^2 / (4 ln 2), and rescaled by one constant per
    (model, dataset, fold) so that it matches mean squared returns on that
    fold's validation block (removes the log-normal / Parkinson bias using
    pre-test data only).
  * Volatility targeting: weight w_t = min(w_max, sigma_target / sigma_t) on a
    long position, cost per unit turnover; Sharpe, max drawdown, realized
    vs target vol, and the Fleming-Kirby-Ostdiek (2001) performance fee for
    switching from a benchmark forecast to the model's.
  * Value-at-Risk: VaR_t = q_alpha * sigma_t, with q_alpha the normal quantile
    or the validation-block quantile of standardized returns (filtered
    historical simulation); Kupiec (1995) and Christoffersen (1998) tests.
Return forecasts (return task)
  * Directional long/short sign(forecast) strategy with costs, as a negative
    control against buy-and-hold.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.paper.data import make_dataset

BARS_PER_YEAR = {"1d": 365, "4h": 6 * 365, "1h": 24 * 365}
PARKINSON = 1.0 / (4.0 * np.log(2.0))


def load_preds(run_dir: str, task: str = "range", split: str = "test") -> pd.DataFrame:
    """Seed-averaged predictions with fold ids, for test or val blocks."""
    from src.paper.evaluate import _to_time
    parts = []
    for path in glob.glob(os.path.join(run_dir, "preds", f"*__{task}__*.npz")):
        ds, tk, gs, model, seed, fold = os.path.basename(path)[:-4].split("__")
        z = np.load(path)
        parts.append(pd.DataFrame({"dataset": ds, "groupset": gs, "model": model, "fold": int(fold[1:]),
                                   "time": _to_time(z[f"{split}_time"]), "pred": z[f"{split}_pred"].astype(float)}))
    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["dataset", "groupset", "model", "fold", "time"], as_index=False).pred.mean()


def next_returns(dataset: str) -> pd.Series:
    """Log return of the bar after each row (what a decision at row t earns)."""
    coin, freq = dataset.split("_")
    f = make_dataset(coin, freq).frame
    return f["y_return"]


def variance_forecasts(test: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Calibrated next-bar variance per row of `test` (columns: ..., var, r)."""
    out = []
    for ds, g in test.groupby("dataset"):
        r = next_returns(ds)
        v = val[val.dataset == ds]
        for key, h in g.groupby(["groupset", "model", "fold"]):
            hv = v[(v.groupset == key[0]) & (v.model == key[1]) & (v.fold == key[2])]
            raw = lambda p: np.exp(p) ** 2 * PARKINSON
            rv = r.reindex(hv.time).to_numpy()
            ok = np.isfinite(rv)
            c = np.mean(rv[ok] ** 2) / np.mean(raw(hv.pred.to_numpy()[ok]))
            # filtered-historical-simulation quantiles of standardized val returns
            zv = rv[ok] / np.sqrt(c * raw(hv.pred.to_numpy()[ok]))
            h = h.assign(var=c * raw(h.pred.to_numpy()), r=r.reindex(h.time).to_numpy(),
                         q01=np.quantile(zv, 0.01), q05=np.quantile(zv, 0.05))
            out.append(h)
    return pd.concat(out, ignore_index=True).dropna(subset=["r"])


def max_drawdown(log_ret: np.ndarray) -> float:
    wealth = np.exp(np.cumsum(log_ret))
    return float(np.max(1 - wealth / np.maximum.accumulate(wealth)))


def fko_fee(rp_model, rp_bench, gamma, gross=True):
    """Annualization left to caller. Fee D (per bar) equating quadratic utility.
    Default: the Fleming et al. (2001) form over the gross return, mean((1 + R) - k (1 + R)^2),
    whose relative risk aversion at R = 0 is gamma. gross=False uses mean(R - k R^2) over the
    net return, a mean-variance utility with risk aversion 2k = gamma / (1 + gamma) only."""
    k = gamma / (2 * (1 + gamma))
    u = (lambda R: np.mean((1 + R) - k * (1 + R) ** 2)) if gross else (lambda R: np.mean(R - k * R ** 2))
    target = u(rp_bench)
    # the gross form is satiated at 1 + R = 1 / (2k), so search a per-bar fee within +-5% there
    lim = 0.05 if gross else 1
    return optimize.brentq(lambda d: u(rp_model - d) - target, -lim, lim)


def fko_fee_parts(rp_model, rp_bench, gamma):
    """First-order split of the gross-return FKO fee (per bar) into a mean and a second-moment part:
    with k = gamma / (2 (1 + gamma)), D ~ dE[R] - k / (1 - 2k) dE[R^2] = dE[R] - (gamma / 2) dE[R^2]."""
    k = gamma / (2 * (1 + gamma))
    d_mean = np.mean(rp_model) - np.mean(rp_bench)
    d_sq = np.mean(rp_model ** 2) - np.mean(rp_bench ** 2)
    return d_mean, -k / (1 - 2 * k) * d_sq


def vol_target(df: pd.DataFrame, freq: str, target_ann=0.40, w_max=2.0, cost=0.0005) -> pd.DataFrame:
    """Per (dataset, groupset, model): vol-targeted long strategy statistics."""
    n = BARS_PER_YEAR[freq]
    tgt = target_ann / np.sqrt(n)
    rows, series = [], {}
    for key, g in df.groupby(["dataset", "groupset", "model"]):
        g = g.sort_values("time")
        w = np.minimum(w_max, tgt / np.sqrt(g["var"].to_numpy()))
        simple = np.expm1(g.r.to_numpy())
        turn = np.abs(np.diff(w, prepend=w[0]))
        rp = w * simple - cost * turn
        series[key] = rp
        rows.append(dict(zip(["dataset", "groupset", "model"], key),
                         ann_ret=np.mean(rp) * n, ann_vol=np.std(rp) * np.sqrt(n),
                         sharpe=np.mean(rp) / np.std(rp) * np.sqrt(n),
                         mdd=max_drawdown(np.log1p(rp)), vol_err=abs(np.std(rp) * np.sqrt(n) - target_ann),
                         turnover=np.mean(turn) * n, lev=np.mean(w), n=len(g)))
    res = pd.DataFrame(rows)
    # buy-and-hold reference on the same timestamps
    for ds, g in df.groupby("dataset"):
        s = np.expm1(g.drop_duplicates("time").sort_values("time").r.to_numpy())
        res.loc[len(res)] = dict(dataset=ds, groupset="-", model="buy&hold", ann_ret=np.mean(s) * n,
                                 ann_vol=np.std(s) * np.sqrt(n), sharpe=np.mean(s) / np.std(s) * np.sqrt(n),
                                 mdd=max_drawdown(np.log1p(s)), vol_err=np.nan, turnover=0.0, lev=1.0, n=len(s))
    return res, series


def var_backtest(df: pd.DataFrame, alpha: float = 0.01, method: str = "fhs") -> pd.DataFrame:
    """method='normal': z = Phi^-1(alpha); 'fhs': empirical alpha-quantile of
    standardized validation returns (per model / dataset / fold)."""
    rows = []
    for key, g in df.groupby(["dataset", "groupset", "model"]):
        g = g.sort_values("time")
        z = g[f"q{int(alpha * 100):02d}"].to_numpy() if method == "fhs" else stats.norm.ppf(alpha)
        hit = (g.r.to_numpy() < z * np.sqrt(g["var"].to_numpy())).astype(int)
        n, x = len(hit), hit.sum()
        p = x / n
        ll = lambda q, k, m: (m - k) * np.log(1 - q) + k * np.log(q) if 0 < q < 1 else 0.0
        lr_uc = -2 * (ll(alpha, x, n) - ll(p, x, n))
        # Christoffersen independence (first-order Markov)
        a, b = hit[:-1], hit[1:]
        n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
        n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
        p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p1 = (n01 + n11) / max(n - 1, 1)
        lr_ind = -2 * (ll(p1, n01 + n11, n - 1) - (ll(p01, n01, n00 + n01) + ll(p11, n11, n10 + n11)))
        rows.append(dict(zip(["dataset", "groupset", "model"], key), n=n, hits=int(x), rate=p,
                         p_uc=1 - stats.chi2.cdf(lr_uc, 1), p_ind=1 - stats.chi2.cdf(lr_ind, 1)))
    return pd.DataFrame(rows)


def directional(run_dir: str, freq: str, cost=0.0005) -> pd.DataFrame:
    """Long/short sign(return forecast) with costs vs buy-and-hold."""
    n = BARS_PER_YEAR[freq]
    test = load_preds(run_dir, task="return")
    test = test[test.dataset.str.endswith(freq)]
    rows = []
    for ds, g in test.groupby("dataset"):
        r = next_returns(ds)
        for key, h in g.groupby(["groupset", "model"]):
            h = h.sort_values("time")
            w = np.sign(h.pred.to_numpy())
            s = np.expm1(r.reindex(h.time).to_numpy())
            rp = w * s - cost * np.abs(np.diff(w, prepend=0))
            rows.append(dict(dataset=ds, groupset=key[0], model=key[1], ann_ret=np.nanmean(rp) * n,
                             sharpe=np.nanmean(rp) / np.nanstd(rp) * np.sqrt(n),
                             hit=np.mean(np.sign(h.pred.to_numpy()) == np.sign(s))))
        s = np.expm1(r.reindex(g.time.drop_duplicates().sort_values()).to_numpy())
        rows.append(dict(dataset=ds, groupset="-", model="buy&hold", ann_ret=np.nanmean(s) * n,
                         sharpe=np.nanmean(s) / np.nanstd(s) * np.sqrt(n), hit=np.nan))
    return pd.DataFrame(rows)
