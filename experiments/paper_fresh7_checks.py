"""
Checks for the seventh external review (Google Deep Review, October 2026).

    .venv/bin/python -W ignore experiments/paper_fresh7_checks.py [name ...]

1. rmsd_coins: rolling 30-day tracking error (RMSD_30) of the volatility-targeted positions per
   coin, for every model of the economic table against HAR, with a Wilcoxon signed-rank test over
   the 15 coins (Table 13 reports the RMSD pooled over coins and dates).
2. har_trainval: HAR fitted on the training block (as in the paper) against HAR refitted on
   training plus validation block before each test block; test MSE ratio per frequency.
3. pool_short: per-coin MSE ratio pooled / single-coin of the same network and inputs (seed
   ensembles), for the four coins whose samples start in 2023 against the other eleven.
Outputs: docs/paper/manuscript/tables/f7_rmsd_coins.csv, f7_har_trainval.csv, f7_pool_short.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from paper_tables_final import (COMBOS, COMBOS_BENCH, ECON, ECON_RV, OUT_T, POOL, _add_combos,  # noqa: E402
                                _harrv_preds, label, run_dir, setting)

SHORT = ["dot", "trx", "bch", "xlm"]         # 4-hourly samples start in mid-2023 (Table 3)


def rmsd_coins(freq="4h", target=0.40, days=30):
    from src.paper.economic import BARS_PER_YEAR, load_preds, variance_forecasts, vol_target
    dirs = [run_dir(f"final_{x}_{freq}") for x in ("single", "pool", "harx", "har_season", "lgbm", "calendar_pool")]
    keep = lambda d: d[[(g, m) in ECON for g, m in zip(d.groupset, d.model)]]
    test = pd.concat([keep(load_preds(d)) for d in dirs if d])
    val = pd.concat([keep(load_preds(d, split="val")) for d in dirs if d])
    rt, rv = _harrv_preds(test, val, freq)
    test, val = pd.concat([test, rt], ignore_index=True), pd.concat([val, rv], ignore_index=True)
    V = _add_combos(variance_forecasts(test, val), variance_forecasts(val, val))
    _, series = vol_target(V, freq)
    n = BARS_PER_YEAR[freq]
    win = int(n / 365 * days)
    times = {k: g.sort_values("time").time.to_numpy() for k, g in V.groupby(["dataset", "groupset", "model"])}
    per = {}
    for (ds, gs, m), rp in series.items():
        rv30 = pd.Series(rp, index=times[(ds, gs, m)]).rolling(win).std() * np.sqrt(n)
        per[(gs, m, ds)] = float(np.sqrt(np.nanmean((rv30.to_numpy() - target) ** 2)))
    s = pd.Series(per)
    har = s.xs(("P", "har"), level=[0, 1])
    rows = []
    for gs, m in ECON + ECON_RV + [(g, "combo:" + x) for g, x in COMBOS + COMBOS_BENCH]:
        if (gs, m) == ("P", "har") or (gs, m) not in s.droplevel(2).index:
            continue
        x = s.xs((gs, m), level=[0, 1]).reindex(har.index)
        name = ("HAR + " + label(gs, m[6:])) if m.startswith("combo:") else label(gs, m)
        rows.append(dict(model=name, rmsd_mean=x.mean(), har_mean=har.mean(), lower=int((x < har).sum()),
                         n=len(x), p_wilcoxon=wilcoxon(x - har).pvalue))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "f7_rmsd_coins.csv", index=False)
    print(t.round(4).to_string(index=False))


def har_trainval():
    from paper_round4_checks import COINS, START
    from src.paper.baselines import HAR_LAGS, _har_design, _ols
    from src.paper.data import make_dataset, walk_forward_folds
    rows = []
    for f in ("4h", "1h", "1d"):
        for coin in COINS:
            d = make_dataset(coin, f, start=START.get(f))
            fr = d.frame
            y = fr["y_range"].to_numpy(float)
            H = _har_design(fr["log_range"].to_numpy(float), HAR_LAGS[f])
            for k, fo in enumerate(walk_forward_folds(fr.index, 4, 135, 180)):
                te = fo["test"]
                se = {}
                for name, tr in (("train", fo["train"]), ("train+val", np.r_[fo["train"], fo["val"]])):
                    ok = tr[~np.isnan(H[tr]).any(1)]
                    b = _ols(H[ok], y[ok])
                    se[name] = ((b[0] + H[te] @ b[1:] - y[te]) ** 2).sum()
                rows.append(dict(freq=f, coin=coin, fold=k, n=len(te), **se))
    t = pd.DataFrame(rows).groupby(["freq", "coin"])[["train", "train+val"]].sum()
    r = (t["train+val"] / t["train"]).rename("ratio").reset_index()
    out = r.groupby("freq").ratio.agg(mean="mean", min="min", max="max",
                                      lower=lambda x: int((x < 1).sum())).reset_index()
    out.to_csv(OUT_T / "f7_har_trainval.csv", index=False)
    print(out.round(4).to_string(index=False))


def pool_short():
    from src.paper.evaluate import seed_mean
    rows = []
    for f in ("4h", "1h", "1d"):
        df = seed_mean(setting(f))
        df = df[df.split == "test"] if "split" in df else df
        mse = df.assign(se=(df.pred - df.y) ** 2).groupby(["groupset", "model", "dataset"]).se.mean()
        for (gs, m), g in mse.groupby(level=[0, 1]):
            if not m.endswith(POOL) or (gs, m[:-len(POOL)]) not in mse.droplevel(2).index:
                continue
            r = (g.droplevel([0, 1]) / mse.loc[(gs, m[:-len(POOL)])]).dropna()
            coin = r.index.str.split("_").str[0]
            short, rest = r[coin.isin(SHORT)], r[~coin.isin(SHORT)]
            rows.append(dict(freq=f, inputs=gs, model=label(gs, m[:-len(POOL)]),
                             short_mean=short.mean(), short_lower=int((short < 1).sum()), short_n=len(short),
                             rest_mean=rest.mean(), rest_lower=int((rest < 1).sum()), rest_n=len(rest)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "f7_pool_short.csv", index=False)
    print(t.round(4).to_string(index=False))
    print(t.groupby("freq")[["short_mean", "rest_mean"]].agg(["mean", "max"]).round(4))
    print(t.groupby("freq")[["short_lower", "short_n", "rest_lower", "rest_n"]].sum())


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for name in sys.argv[1:] or ["rmsd_coins", "har_trainval", "pool_short"]:
        globals()[name]()
