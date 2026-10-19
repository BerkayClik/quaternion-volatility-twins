"""
Analysis of the internal-review round-7 Colab runs (October 2026).

    .venv/bin/python -W ignore experiments/paper_round7_colab.py [name[:arg] ...]

1. seeds10: single-coin daily twins (PMSE) with ten seeds (seeds 0-2 from final_single_1d, 3-9 from
   final_seeds10_single_1d): ensemble ratio (mean of per-coin MSE ratios, coins better, cross-coin DM
   on the coin-averaged loss), per-seed ratios of coin-averaged MSE and a t-test of their logs.
2. lr:<setting> (single_1d or pool_4h): learning rates 3e-4 / 1e-3 / 3e-3 (seeds 0-2); per model the
   validation and test MSE relative to HAR (means of per-coin ratios), median best epoch, the twin
   ratio at each rate and with the rate chosen per model on the validation blocks.
3. w2:<freq> (4h or 1d): second test window (2023-10-29/30 to 2025-04-21/22, configs w2_pool_*):
   twin ratios of the pooled networks (PM, PMSE) with cross-coin DM and the +-1% equivalence interval,
   the calendar networks against HAR, HAR-S, HAR-S with weekday x time-of-day interactions, pooled
   LightGBM-S and LightGBM-S with lagged inputs (both refitted here on the same folds), and the same
   numbers in the main window for the same coins and seeds.
Outputs: docs/paper/manuscript/tables/i7c_<name>.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from paper_round7_checks import _cfg, _interactions, _load, _nw_lrv  # noqa: E402
from paper_tables_final import OUT_T, POOL, label  # noqa: E402

TWINS = [("lstm", "qlstm"), ("transformer", "qtransformer"), ("hl_meta", "hq_meta"), ("ht_meta", "hqt_meta"),
         ("hl_gattn", "hq_gattn"), ("ht_concat", "hqt_concat")]


def _ens(df):
    """Seed-ensemble forecasts: (dataset, groupset, model, time) -> pred, y."""
    return df.groupby(["dataset", "groupset", "model", "time"], as_index=False).agg(y=("y", "first"),
                                                                                 pred=("pred", "mean"))


def _mse(e):
    return e.assign(se=(e.pred - e.y) ** 2).groupby(["groupset", "model", "dataset"]).se.mean()


def _cross_dm(e, gs, a, b):
    """Cross-coin DM of a vs b (coin-averaged squared-error differential on aligned dates) and the
    +-1% equivalence interval of the ratio of coin-averaged losses (as paper_review_checks)."""
    P = {m: e[(e.groupset == gs) & (e.model == m)].assign(se=lambda x: (x.pred - x.y) ** 2)
         .pivot_table(index="time", columns="dataset", values="se") for m in (a, b)}
    La, Lb = P[a].mean(1), P[b].mean(1)
    idx = La.dropna().index.intersection(Lb.dropna().index)
    d = (La[idx] - Lb[idx]).to_numpy()
    lag = int(len(d) ** (1 / 3))
    se = np.sqrt(_nw_lrv(d, lag) / len(d))
    r = 1 + d.mean() / Lb[idx].mean()
    half = 1.645 * se / Lb[idx].mean()
    return dict(ratio_pooled=r, dm_p=2 * stats.norm.sf(abs(d.mean() / se)), ci_lo=r - half, ci_hi=r + half,
                equiv=bool(r - half > 0.99 and r + half < 1.01))


def seeds10():
    keep = lambda g, m: g == "PMSE" and any(m in t for t in TWINS)
    df = pd.concat([_load(["final_single_1d"], keep, seeds=[0, 1, 2]),
                    _load(["final_seeds10_single_1d"], keep)], ignore_index=True)
    rows = []
    for real, quat in TWINS[:4]:
        sub = df[df.model.isin([real, quat])]
        per_seed = []
        for s, g in sub.groupby("seed"):
            m = _mse(g.rename(columns={}))
            per_seed.append(m.xs(("PMSE", quat)).mean() / m.xs(("PMSE", real)).mean())
        per_seed = np.array(per_seed)
        out = {}
        for tag, seeds in (("3", [0, 1, 2]), ("10", list(range(10)))):
            e = _ens(sub[sub.seed.isin(seeds)])
            m = _mse(e)
            r = m.xs(("PMSE", quat)) / m.xs(("PMSE", real))
            dm = _cross_dm(e, "PMSE", quat, real)
            out.update({f"ratio_{tag}": r.mean(), f"better_{tag}": int((r < 1).sum()), f"dm_p_{tag}": dm["dm_p"],
                        f"ci_lo_{tag}": dm["ci_lo"], f"ci_hi_{tag}": dm["ci_hi"]})
        rows.append(dict(pair=f"{label('PMSE', quat)} / {label('PMSE', real)}", **out,
                         n_seeds=len(per_seed), seed_min=per_seed.min(), seed_max=per_seed.max(),
                         seeds_worse=int((per_seed > 1).sum()),
                         ttest_log_p=stats.ttest_1samp(np.log(per_seed), 0).pvalue,
                         sign_p=stats.binomtest(int((per_seed > 1).sum()), len(per_seed)).pvalue))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "i7c_seeds10.csv", index=False)
    print(t.round(4).to_string(index=False))


LR_RUNS = {"single_1d": {3e-4: "final_lr3e4_single_1d", 1e-3: "final_single_1d", 3e-3: "final_lr3e3_single_1d",
                         1e-2: "final_lr1e2_single_1d"},       # 1e-2: internal review round 8 (Colab r9)
           "pool_4h": {3e-4: "final_lr3e4_pool_4h", 1e-3: "final_pool_4h", 3e-3: "final_lr3e3_pool_4h",
                       1e-2: "final_lr1e2_pool_4h"}}


def _best_epochs(name, models, pooled):
    import glob
    import json
    from paper_tables_final import run_dir
    out = {}
    for path in glob.glob(os.path.join(run_dir(name), "preds", "*__PMSE__*.npz")):
        ds, _, _, model, seed, _ = os.path.basename(path)[:-4].split("__")
        if model not in models or int(seed[1:]) > 2 or (pooled and not ds.startswith("btc_")):
            continue
        out.setdefault(model, []).append(json.loads(str(np.load(path)["meta"]))["best_epoch"])
    return {m: float(np.median(v)) for m, v in out.items()}


def lr(setting="single_1d"):
    pooled = setting.startswith("pool")
    sfx = POOL if pooled else ""
    pairs = TWINS[:4] + ([TWINS[4]] if pooled else [])
    models = {m + sfx for p in pairs for m in p}
    freq = setting[-2:]
    har = _load([f"final_single_{freq}"], lambda g, m: g == "P" and m == "har", splits=("test", "val"))
    har = {s: _mse(_ens(g)).xs(("P", "har")) for s, g in har.groupby("split")}
    res = {}
    for lrate, name in LR_RUNS[setting].items():
        df = _load([name], lambda g, m: g == "PMSE" and m in models, splits=("test", "val"), seeds=[0, 1, 2])
        ep = _best_epochs(name, models, pooled)
        for split, g in df.groupby("split"):
            m = _mse(_ens(g))
            for mod in models:
                r = m.xs(("PMSE", mod)) / har[split]
                res.setdefault((mod, lrate), {})[split] = r.mean()
                res[(mod, lrate)][f"{split}_coins"] = m.xs(("PMSE", mod))
            for mod in models:
                res[(mod, lrate)]["best_epoch"] = ep.get(mod)
    rows = []
    for (mod, lrate), v in sorted(res.items()):
        rows.append(dict(setting=setting, model=mod, lr=lrate, val_vs_har=v["val"], test_vs_har=v["test"],
                         best_epoch=v["best_epoch"]))
    per = pd.DataFrame(rows)
    sel = per.loc[per.groupby("model").val_vs_har.idxmin()].set_index("model")
    twin_rows = []
    for real, quat in pairs:
        q, r_ = quat + sfx, real + sfx
        row = dict(setting=setting, pair=f"{label('PMSE', q)} / {label('PMSE', r_)}")
        for lrate in LR_RUNS[setting]:
            row[f"lr{lrate:g}"] = (res[(q, lrate)]["test_coins"] / res[(r_, lrate)]["test_coins"]).mean()
        lq, lr_ = sel.loc[q, "lr"], sel.loc[r_, "lr"]
        row.update(lr_q=lq, lr_r=lr_,
                   selected=(res[(q, lq)]["test_coins"] / res[(r_, lr_)]["test_coins"]).mean(),
                   selected_better=int((res[(q, lq)]["test_coins"] < res[(r_, lr_)]["test_coins"]).sum()))
        twin_rows.append(row)
    tw = pd.DataFrame(twin_rows)
    per.to_csv(OUT_T / f"i7c_lr_models_{setting}.csv", index=False)
    tw.to_csv(OUT_T / f"i7c_lr_twins_{setting}.csv", index=False)
    print(per.round(4).to_string(index=False))
    print(tw.round(4).to_string(index=False))


def _refit_benchmarks(cfgname):
    """HAR-S with interactions and pooled LightGBM-S with lagged inputs on the folds of a config;
    returns dataset -> DataFrame(time x {hars_int, lgbms_lags})."""
    import lightgbm  # noqa: F401
    from src.paper.baselines import HAR_LAGS, _har_design, _ols, predict_baseline_pooled
    from src.paper.data import make_dataset, walk_forward_folds
    cfg = _cfg(cfgname)
    freq = cfg["datasets"][0].split("_")[1]
    fc, win = cfg["folds"], cfg["window"][freq]
    fr = {}
    for ds in cfg["datasets"]:
        d = make_dataset(ds.split("_")[0], freq, start=cfg.get("start"))
        fr[ds] = (d, walk_forward_folds(d.frame.index, fc["n"], fc["test_days"], fc["val_days"], end=fc.get("end")))
    out = {ds: {} for ds in fr}
    if freq != "1d":
        for ds, (d, folds) in fr.items():
            y = d.frame["y_range"].to_numpy(float)
            D = np.column_stack([_har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[freq]),
                                 _interactions(d.frame.index, d.step)])
            s = pd.Series(np.nan, index=d.frame.index)
            for fo in folds:
                tr, te = fo["train"], fo["test"]
                te = te[te >= win - 1]
                ok = tr[~np.isnan(D[tr]).any(1)]
                b = _ols(D[ok], y[ok])
                s.iloc[te] = b[0] + D[te] @ b[1:]
            out[ds]["hars_int"] = s.dropna()
    groups = cfg["groupsets"]["PM"]
    dss = list(fr)
    Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS[freq][1:]])
          for X in (fr[ds][0].X(groups) for ds in dss)]
    parts = {ds: [] for ds in dss}
    for k in range(fc["n"]):
        res = predict_baseline_pooled("lgbms", [fr[ds][0].frame for ds in dss], "range",
                                      [fr[ds][1][k] for ds in dss], win, freq, Xl, seed=0)
        for ds, r in zip(dss, res):
            idx = fr[ds][0].frame.index
            parts[ds].append(pd.Series(r["test_pred"], index=idx[r["test_rows"]]))
    for ds in dss:
        out[ds]["lgbms_lags"] = pd.concat(parts[ds])
    return out


def w2(freq="1d"):
    cfg = _cfg(f"w2_pool_{freq}")
    coins = cfg["datasets"]
    models = {m + POOL for p in TWINS for m in p} | {"lgbms-pool", "har", "hars"}
    rows, bench = [], []
    for window, names in (("w2", [f"final_w2_pool_{freq}"]),
                          ("main", [f"final_pool_{freq}", f"final_calendar_pool_{freq}", f"final_single_{freq}",
                                    f"final_har_season_{freq}", f"final_lgbm_{freq}"])):
        df = _load(names, lambda g, m: m in models, seeds=[0, 1, 2])
        df = df[df.dataset.isin(coins)]
        if window == "main":   # main-window benchmarks live under other groupsets; relabel to w2's
            df.loc[df.model.isin(["har"]), "groupset"] = "PM"
            df.loc[df.model.isin(["hars"]), "groupset"] = "PM"
        e = _ens(df)
        m = _mse(e)
        for gs in ("PM", "PMSE"):
            for real, quat in TWINS:
                q, r_ = quat + POOL, real + POOL
                if (gs, q) not in m.droplevel(2).index or (gs, r_) not in m.droplevel(2).index:
                    continue
                r = m.xs((gs, q)) / m.xs((gs, r_))
                rows.append(dict(window=window, inputs=gs, pair=f"{label(gs, q)} / {label(gs, r_)}",
                                 ratio=r.mean(), better=int((r < 1).sum()), n=len(r), **_cross_dm(e, gs, q, r_)))
        # benchmarks and calendar networks against HAR
        refit = _refit_benchmarks(f"w2_pool_{freq}" if window == "w2" else f"lgbm_{freq}")
        har = m.xs(("PM", "har"))
        y = e[(e.groupset == "PM") & (e.model == "har")].set_index(["dataset", "time"]).y
        extra = {}
        for k in ("hars_int", "lgbms_lags"):
            v = []
            for ds in coins:
                if k not in refit[ds]:
                    continue
                yy = y.xs(ds)
                p = refit[ds][k].reindex(yy.index)
                v.append(pd.Series({ds: np.nanmean((p.to_numpy() - yy.to_numpy()) ** 2)}))
            if v:
                extra[k] = pd.concat(v)
        cands = {"HAR-S": m.xs(("PM", "hars")) if ("PM", "hars") in m.droplevel(2).index else m.xs(("PMSE", "hars")),
                 "LightGBM-S (pooled)": m.xs(("PM", "lgbms-pool")),
                 **({"HAR-S x (interactions)": extra["hars_int"]} if "hars_int" in extra else {}),
                 "LightGBM-S lags (pooled)": extra["lgbms_lags"]}
        for gs in ("PM", "PMSE", "PMC"):
            for mod in ("hl_gattn-pool", "lstm-pool", "transformer-pool", "ht_concat-pool"):
                if (gs, mod) in m.droplevel(2).index:
                    cands[label(gs, mod) + f", {gs}"] = m.xs((gs, mod))
        net = m.xs(("PMC", "hl_gattn-pool"))
        for name, v in cands.items():
            v = v.reindex(har.index)
            bench.append(dict(window=window, model=name, vs_har=(v / har).mean(), better_than_har=int((v < har).sum()),
                              cal_net_vs=(net / v).mean(), cal_net_better=int((net < v).sum()), n=len(har)))
    t, b = pd.DataFrame(rows), pd.DataFrame(bench)
    t.to_csv(OUT_T / f"i7c_w2_twins_{freq}.csv", index=False)
    b.to_csv(OUT_T / f"i7c_w2_bench_{freq}.csv", index=False)
    print(t.round(4).to_string(index=False))
    print(b.round(4).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for arg in sys.argv[1:] or ["seeds10", "lr:single_1d", "w2:1d"]:
        name, *a = arg.split(":")
        globals()[name](*a)
