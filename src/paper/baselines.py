"""
Non-neural baselines, fitted on the same folds as the networks.

  persistence  range: y_hat = current log range;  return: y_hat = 0
  har          range only: OLS on daily/weekly/monthly mean log range (Corsi 2009)
  mean         train-fold mean of the target
  harx         HAR plus the current bar's input features
  hars, harxs  HAR / HAR-X plus calendar dummies of the target bar: time of day (intraday
               bars) and day of week, so the benchmark can represent intraday and weekly
               seasonality (cf. Andersen and Bollerslev 1997)
  ridge        ridge regression on the flattened input window (alpha picked on val)
  lgbm, lgbms  gradient-boosted trees (LightGBM) on the HAR regressors and the current bar's
               input features (lgbms: plus the calendar of the target bar); early stopping
               on the validation block
  pooled HAR   predict_baseline_pooled: one OLS on the stacked, per-coin standardised
               HAR design of all coins (panel HAR; Bollerslev et al. 2018)
  garch, gjr   range only: GARCH(1,1) / GJR-GARCH(1,1,1) with Student-t errors fitted on
               train-fold returns (Bollerslev 1986; Glosten et al. 1993); the one-step
               conditional volatility is filtered forward with fixed parameters and mapped
               to the log-range scale by OLS on the train rows
"""

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import Ridge

# HAR horizons in bars: (short, medium, long)
BAR = {"1d": "1D", "4h": "4h", "1h": "1h"}
HAR_LAGS = {"1d": (1, 7, 30), "4h": (1, 6, 42), "1h": (1, 24, 168)}


def _har_design(log_range: np.ndarray, lags) -> np.ndarray:
    cols = [np.convolve(log_range, np.ones(k) / k, mode="full")[: len(log_range)] for k in lags]
    X = np.column_stack(cols)
    X[: max(lags) - 1] = np.nan
    return X


def _calendar(index, step) -> np.ndarray:
    """Dummies (one level dropped) for the time of day and the day of week of bar t+1.
    The calendar of the next bar is known at the end of bar t."""
    nxt = index + step
    cols = []
    tod = nxt.hour * 60 + nxt.minute
    for v in np.unique(tod)[1:]:
        cols.append((tod == v).astype(float))
    for d in range(1, 7):
        cols.append((nxt.dayofweek == d).astype(float))
    return np.column_stack(cols)


def _design(name, frame, freq, X=None):
    """HAR-type regressors of every row for the HAR family of baselines."""
    H = _har_design(frame["log_range"].to_numpy(float), HAR_LAGS[freq])
    parts = [H]
    if name.startswith(("harx", "lgbm")):
        parts.append(X)
    if name.endswith("s"):
        parts.append(_calendar(frame.index, pd.Timedelta(BAR[freq])))
    return np.column_stack(parts)


LGBM_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   n_estimators=3000, random_state=0, verbose=-1, n_jobs=4)


def _lgbm(Xtr, ytr, Xva, yva, seed=0):
    """LightGBM regressor with early stopping (patience 100) on the validation block.
    `seed` drives the 0.8 row / column subsampling (seed 0 = the original single fit)."""
    import lightgbm as lgb  # optional dependency, only for this baseline
    m = lgb.LGBMRegressor(**{**LGBM_PARAMS, "random_state": seed})
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def _ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _garch_log_sigma(ret: np.ndarray, train_rows: np.ndarray, asymmetric: bool) -> np.ndarray:
    """log sigma_{t+1|t} for every row t (known at the end of bar t)."""
    r = 100.0 * np.nan_to_num(ret)
    res = arch_model(r[train_rows], mean="Constant", vol="GARCH", p=1, o=int(asymmetric), q=1,
                     dist="t").fit(disp="off")
    p = res.params
    mu, omega, alpha, beta = p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"]
    gamma = p["gamma[1]"] if asymmetric else 0.0
    eps = r - mu
    var = np.empty(len(r))
    s2 = np.var(r[train_rows])
    for t in range(len(r)):           # s2 holds sigma^2_t; var[t] stores sigma^2_{t+1}
        s2 = omega + (alpha + gamma * (eps[t] < 0)) * eps[t] ** 2 + beta * s2
        var[t] = s2
    return 0.5 * np.log(var)


def _windows(X, rows, window):
    rows = rows[rows >= window - 1]
    return rows, np.stack([X[t - window + 1: t + 1].ravel() for t in rows])


def predict_baseline(name, frame, task, fold, window, freq, X=None):
    """Return {'test_rows', 'test_pred', 'val_rows', 'val_pred'} like train.fit_predict."""
    y = frame[f"y_{task}"].to_numpy(float)
    tr, va, te = fold["train"], fold["val"], fold["test"]
    sel = lambda rows: rows[rows >= window - 1]          # same sample as the networks
    va, te = sel(va), sel(te)

    if name == "persistence":
        base = frame["log_range"].to_numpy(float) if task == "range" else np.zeros(len(frame))
        return {"test_rows": te, "test_pred": base[te], "val_rows": va, "val_pred": base[va]}

    if name == "mean":
        mu = y[tr].mean()
        return {"test_rows": te, "test_pred": np.full(len(te), mu),
                "val_rows": va, "val_pred": np.full(len(va), mu)}

    if name in ("har", "hars", "harxs"):
        if task != "range":
            raise ValueError("HAR is defined for the range task only")
        H = _design(name, frame, freq, X)
        ok = tr[~np.isnan(H[tr]).any(1)]
        beta = _ols(H[ok], y[ok])
        f = lambda rows: beta[0] + H[rows] @ beta[1:]
        return {"test_rows": te, "test_pred": f(te), "val_rows": va, "val_pred": f(va)}

    if name == "harx":
        # HAR-X: the HAR regressors plus the current bar's features of the
        # input groups (e.g. market, social, sentiment), estimated by OLS.
        if task != "range":
            raise ValueError("HAR-X is defined for the range task only")
        H = np.column_stack([_har_design(frame["log_range"].to_numpy(float), HAR_LAGS[freq]), X])
        ok = tr[~np.isnan(H[tr]).any(1)]
        beta = _ols(H[ok], y[ok])
        f = lambda rows: beta[0] + H[rows] @ beta[1:]
        return {"test_rows": te, "test_pred": f(te), "val_rows": va, "val_pred": f(va)}

    if name in ("lgbm", "lgbms"):
        if task != "range":
            raise ValueError("the boosting baseline is defined for the range task only")
        H = _design(name, frame, freq, X)
        ok = tr[~np.isnan(H[tr]).any(1)]
        m = _lgbm(H[ok], y[ok], H[va], y[va])
        return {"test_rows": te, "test_pred": m.predict(H[te]), "val_rows": va, "val_pred": m.predict(H[va])}

    if name in ("garch", "gjr"):
        if task != "range":
            raise ValueError("GARCH baselines are defined for the range task only")
        z = _garch_log_sigma(frame["ret"].to_numpy(float), tr, asymmetric=name == "gjr")
        beta = _ols(z[tr, None], y[tr])
        f = lambda rows: beta[0] + beta[1] * z[rows]
        return {"test_rows": te, "test_pred": f(te), "val_rows": va, "val_pred": f(va)}

    if name == "ridge":
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = 1.0
        Xs = (X - mu) / sd
        tr_rows, Wtr = _windows(Xs, tr, window)
        _, Wva = _windows(Xs, va, window)
        _, Wte = _windows(Xs, te, window)
        best = None
        for alpha in 10.0 ** np.arange(-2, 5):
            m = Ridge(alpha=alpha).fit(Wtr, y[tr_rows])
            err = np.mean((m.predict(Wva) - y[va]) ** 2)
            if best is None or err < best[0]:
                best = (err, m)
        m = best[1]
        return {"test_rows": te, "test_pred": m.predict(Wte), "val_rows": va, "val_pred": m.predict(Wva)}

    raise ValueError(f"unknown baseline {name}")


def predict_baseline_pooled(name, frames, task, folds, window, freq, Xs, seed=0):
    """Panel version of the HAR family (and of the boosting baseline): per-coin standardised regressors and target
    (train-fold statistics), one OLS on all coins' training rows, forecasts mapped back
    to each coin's scale. Returns one result dict per coin, like predict_baseline."""
    if task != "range":
        raise ValueError("pooled baselines are defined for the range task only")
    designs, stats, rows = [], [], []
    for frame, fold, X in zip(frames, folds, Xs):
        H = _design(name, frame, freq, X)
        y = frame["y_range"].to_numpy(float)
        tr = fold["train"]
        ok = tr[~np.isnan(H[tr]).any(1)]
        mx, sx = H[ok].mean(0), H[ok].std(0)
        sx[sx < 1e-8] = 1.0
        my, sy = y[ok].mean(), y[ok].std()
        designs.append(((H - mx) / sx, (y - my) / sy))
        stats.append((my, sy))
        rows.append(ok)
    A = np.vstack([D[r] for (D, _), r in zip(designs, rows)])
    b = np.concatenate([z[r] for (_, z), r in zip(designs, rows)])
    sel = lambda r: r[r >= window - 1]
    if name.startswith("lgbm"):
        vrows = [sel(f["val"]) for f in folds]
        m = _lgbm(A, b, np.vstack([D[v] for (D, _), v in zip(designs, vrows)]),
                  np.concatenate([z[v] for (_, z), v in zip(designs, vrows)]), seed=seed)
        g = lambda D, r: m.predict(D[r])
    else:
        beta = _ols(A, b)
        g = lambda D, r: beta[0] + D[r] @ beta[1:]
    out = []
    for (D, _), (my, sy), fold in zip(designs, stats, folds):
        va, te = sel(fold["val"]), sel(fold["test"])
        f = lambda r: my + sy * g(D, r)
        out.append({"test_rows": te, "test_pred": f(te), "val_rows": va, "val_pred": f(va)})
    return out
