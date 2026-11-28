"""Round-9 checks (internal review round 8): local computations on saved forecasts.

  clark_west   nested test of the LunarCrush groups in HAR-X (PM inside PMSE), all frequencies
               -> tables/r9_clark_west.csv
  trading      4-hourly volatility targeting under alternative trading rules (10 bp costs, no leverage,
               daily rebalancing, turnover against the drifted weight), every forecast level-calibrated
               -> tables/r9_trading.csv
  nw_lags      cross-coin DM tests of the best calendar network against the strongest benchmarks with
               Newey-West bandwidths covering the daily and weekly cycles -> tables/r9_nw_lags.csv
  har_rmse     per-coin RMSE of HAR's log-range forecasts, to express MSE ratios in volatility units
               -> tables/r9_har_rmse.csv
  lgbm_tune    lagged pooled LightGBM-S with hyperparameters chosen on the validation blocks (seed 0)
               -> results/r9_local/final_lgbmtuned_<freq>_r9/preds, tables/r9_lgbm_tune_<freq>.csv
  tex          LaTeX table of the trading rules (tables/r9_trading.tex)

Run:  .venv/bin/python experiments/paper_round9_checks.py clark_west
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_round7_checks import _load, _nw_lrv  # noqa: E402

OUT_T = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "docs", "paper", "manuscript", "tables")
FREQS = ("1h", "4h", "1d")


def _aligned(df, a, b):
    """Per coin: y, forecast of a, forecast of b on common test times (seed means)."""
    e = df.groupby(["dataset", "groupset", "model", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
    out = {}
    for ds, g in e.groupby("dataset"):
        A = g[(g.groupset == a[0]) & (g.model == a[1])].set_index("time")
        B = g[(g.groupset == b[0]) & (g.model == b[1])].set_index("time")
        t = A.index.intersection(B.index).sort_values()
        out[ds] = (A.loc[t, "y"], A.loc[t, "pred"], B.loc[t, "pred"])
    return out


def clark_west():
    """Clark and West (2007) adjusted-MSPE test of the small model PM-HAR-X (price and market inputs)
    against the larger PMSE-HAR-X that adds the LunarCrush social and sentiment groups. Per coin
    f_t = e_small^2 - (e_large^2 - (p_small - p_large)^2); the cross-coin version averages f_t over coins on
    each timestamp (as evaluate.dm_cross) and applies a one-sided Newey-West t-test (H1: the larger
    model is better). The plain DM statistic of the MSE difference is given for comparison."""
    small, large = ("PM", "harx"), ("PMSE", "harx")
    rows = []
    for f in FREQS:
        df = _load([f"final_harx_{f}"], lambda g, m: (g, m) in (small, large))
        per, cw, dd = [], [], []
        for ds, (y, ps, pl) in _aligned(df, small, large).items():
            es, el = (y - ps) ** 2, (y - pl) ** 2
            fcw = es - (el - (ps - pl) ** 2)
            per.append(dict(dataset=ds, ratio=el.mean() / es.mean(),
                            t_cw=fcw.mean() / np.sqrt(_nw_lrv(fcw.to_numpy(), int(len(fcw) ** (1 / 3))) / len(fcw))))
            cw.append(fcw.rename(ds))
            dd.append((es - el).rename(ds))
        per = pd.DataFrame(per)
        for name, parts in (("cw", cw), ("dm", dd)):
            d = pd.concat(parts, axis=1).mean(axis=1).sort_index().to_numpy()
            lag = int(len(d) ** (1 / 3))
            t = d.mean() / np.sqrt(_nw_lrv(d, lag) / len(d))
            if name == "cw":
                t_cw, p_cw = t, float(stats.norm.sf(t))
            else:
                t_dm, p_dm = t, float(2 * stats.norm.sf(abs(t)))
        rows.append(dict(freq=f, mean_ratio_large_small=per.ratio.mean(), better=int((per.ratio < 1).sum()),
                         n=len(per), t_cw=t_cw, p_cw_one=p_cw, coins_cw_sig=int((per.t_cw > 1.645).sum()),
                         t_dm=t_dm, p_dm_two=p_dm))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_T, "r9_clark_west.csv"), index=False)
    print(t.round(4).to_string(index=False))
    return t


def _prepare_econ(freq="4h"):
    """Test and validation variance forecasts of the economic evaluation, as paper_round8_checks.econ_cal
    (Table 13 models, the lagged LightGBM-S and the HAR combinations)."""
    import paper_tables_final as T
    from paper_round8_checks import LGBM_LAG, LGBM_LAG_DIR
    from src.paper.economic import load_preds, variance_forecasts
    econ = T.ECON + [LGBM_LAG]
    T.COMBOS = T.COMBOS + [("PMC", "hl_gattn-pool"), LGBM_LAG]
    dirs = [T.run_dir(f"final_{x}_{freq}") for x in ("single", "pool", "harx", "har_season", "lgbm", "calendar_pool")]
    dirs.append(LGBM_LAG_DIR.format(f=freq))
    keep = lambda d: d[[(g, m) in econ for g, m in zip(d.groupset, d.model)]]
    test = pd.concat([keep(load_preds(d)) for d in dirs if d])
    val = pd.concat([keep(load_preds(d, split="val")) for d in dirs if d])
    Vv = variance_forecasts(val, val)
    V = T._add_combos(variance_forecasts(test, val), Vv)
    return V, T._add_combos(Vv, Vv)


def _strategy(var, r, n, target=0.40, w_max=2.0, cost=0.0005, hold=1, drift=False):
    """Returns of the volatility-targeted long position. The weight min(w_max, target / sigma) is reset every
    `hold` bars from the forecast of that bar; costs are charged on the change of the weight, measured
    against the previous target weight or, with drift=True, against the weight after the last bar's return
    has moved the position (w (1 + r) / (1 + w r))."""
    tgt = target / np.sqrt(n)
    w = np.minimum(w_max, tgt / np.sqrt(var))
    if hold > 1:
        idx = (np.arange(len(w)) // hold) * hold
        w = w[idx]
    simple = np.expm1(r)
    prev = np.concatenate([[w[0]], w[:-1]])
    if drift:
        ps = np.concatenate([[0.0], simple[:-1]])
        prev = prev * (1 + ps) / (1 + prev * ps)
    turn = np.abs(w - prev)
    return w * simple - cost * turn, turn


def trading(reps=500, seed=0):
    """Every forecast is level-calibrated on validation data under each rule (one constant per model, coin and
    fold, set so that the strategy's validation volatility is 40%), then evaluated on the test period.
    Reports per rule and model: Sharpe ratio, volatility error, turnover and the FKO fee (gamma = 10) against the
    level-calibrated HAR with a moving-block bootstrap 90% interval; buy-and-hold for reference."""
    from scipy import optimize
    from src.paper.economic import BARS_PER_YEAR, fko_fee
    freq = "4h"
    n = BARS_PER_YEAR[freq]
    V, Vv = _prepare_econ(freq)
    rules = {"main (5 bp, w<=2)": dict(), "10 bp": dict(cost=0.001), "no leverage (w<=1)": dict(w_max=1.0),
             "daily rebalancing": dict(hold=6), "drift-aware turnover": dict(drift=True),
             "10 bp, w<=1, daily": dict(cost=0.001, w_max=1.0, hold=6)}
    vv = {k: g.sort_values("time") for k, g in Vv.groupby(["groupset", "model", "dataset", "fold"])}
    rows = []
    for rname, kw in rules.items():
        series, stats_ = {}, []
        for (gs, m, ds), g in V.groupby(["groupset", "model", "dataset"]):
            parts = []
            for fold, h in g.sort_values("time").groupby("fold"):
                v = vv[(gs, m, ds, fold)]
                rv, var_v = v.r.to_numpy(), v["var"].to_numpy()

                def gap(logk):
                    rp, _ = _strategy(np.exp(logk) * var_v, rv, n, **kw)
                    return np.std(rp) * np.sqrt(n) - 0.40
                try:
                    k = np.exp(optimize.brentq(gap, -8, 8))
                except ValueError:                       # target unreachable under the cap: closest constant
                    k = np.exp(optimize.minimize_scalar(lambda x: abs(gap(x)), bounds=(-8, 8), method="bounded").x)
                parts.append(h.assign(var=k * h["var"].to_numpy()))
            h = pd.concat(parts).sort_values("time")
            rp, turn = _strategy(h["var"].to_numpy(), h.r.to_numpy(), n, **kw)
            series[(ds, gs, m)] = rp
            stats_.append(dict(dataset=ds, groupset=gs, model=m, sharpe=rp.mean() / rp.std() * np.sqrt(n),
                               vol_err=abs(rp.std() * np.sqrt(n) - 0.40), turnover=turn.mean() * n))
        st = pd.DataFrame(stats_)
        coins = sorted(st.dataset.unique())
        T_ = len(series[next(iter(series))])
        rng = np.random.default_rng(seed)
        Lb = max(2, int(T_ ** (1 / 3)))
        nb = -(-T_ // Lb)
        draws = [(rng.integers(0, T_ - Lb + 1, nb)[:, None] + np.arange(Lb)).ravel()[:T_] for _ in range(reps)]
        fee = lambda a, b, ix=slice(None): np.mean([fko_fee(series[(c,) + a][ix], series[(c,) + b][ix], 10) * n * 1e4
                                                   for c in coins])
        bh = np.mean([np.expm1(g.drop_duplicates("time").sort_values("time").r.to_numpy()).mean()
                      / np.expm1(g.drop_duplicates("time").sort_values("time").r.to_numpy()).std() * np.sqrt(n)
                      for _, g in V.groupby("dataset")])
        for (gs, m), g in st.groupby(["groupset", "model"], sort=False):
            rec = dict(rule=rname, groupset=gs, model=m, sharpe=g.sharpe.mean(), vol_err=g.vol_err.mean(),
                       turnover=g.turnover.mean(), buyhold_sharpe=bh)
            if (gs, m) != ("P", "har"):
                bs = np.array([fee((gs, m), ("P", "har"), ix) for ix in draws])
                rec.update(fee=fee((gs, m), ("P", "har")), fee_lo=np.percentile(bs, 5), fee_hi=np.percentile(bs, 95))
            rows.append(rec)
        print(rname, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_T, "r9_trading.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(t.round(3).to_string(index=False))
    return t


def nw_lags():
    """Cross-coin DM statistic (loss differential averaged over coins per timestamp, as evaluate.dm_cross) of
    the pooled calendar H-LSTM (PMC, seeds 0-2) against the three-seed lagged LightGBM-S and HAR-S x, with
    the default bandwidth floor(n^(1/3)) and with bandwidths of one day and one week of bars."""
    from paper_round7_checks import CAL_NET
    from paper_round8_checks import HARSX, LGBM_LAG
    lags = {"1h": (24, 168), "4h": (6, 42), "1d": (7,)}
    rows = []
    for f in FREQS:
        names = [f"final_calendar_pool_{f}", f"final_lgbmlags_{f}", f"final_harsx_{f}"]
        df = _load(names, lambda g, m: (g, m) in {CAL_NET, LGBM_LAG, HARSX})
        df = df[~((df.groupset == CAL_NET[0]) & (df.model == CAL_NET[1]) & (df.seed > 2))]
        for b in (LGBM_LAG, HARSX):
            al = _aligned(df, CAL_NET, b)
            if not al:
                continue
            d = pd.concat([((pa - y) ** 2 - (pb - y) ** 2).rename(ds) for ds, (y, pa, pb) in al.items()],
                          axis=1).mean(axis=1).sort_index().to_numpy()
            for lag in (int(np.floor(len(d) ** (1 / 3))),) + lags[f]:
                t = d.mean() / np.sqrt(_nw_lrv(d, lag) / len(d))
                rows.append(dict(freq=f, b=f"{b[1]} {b[0]}", lag=lag, n=len(d), dm=t, p=float(2 * stats.norm.sf(abs(t)))))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_T, "r9_nw_lags.csv"), index=False)
    print(t.round(4).to_string(index=False))
    return t


def har_rmse():
    """Median over coins of HAR's test RMSE (log range) and what an MSE ratio of 0.99 or 0.94 means for the
    typical forecast error, in log points (= relative error of the volatility level)."""
    rows = []
    for f in FREQS:
        df = _load([f"final_single_{f}"], lambda g, m: (g, m) == ("P", "har"))
        e = df.groupby(["dataset", "time"], as_index=False).agg(y=("y", "first"), p=("pred", "mean"))
        rmse = e.groupby("dataset").apply(lambda g: np.sqrt(np.mean((g.y - g.p) ** 2)))
        med = float(rmse.median())
        rows.append(dict(freq=f, rmse_median=med, rmse_min=float(rmse.min()), rmse_max=float(rmse.max()),
                         drop_ratio_099=med * (1 - np.sqrt(0.99)), drop_ratio_094=med * (1 - np.sqrt(0.94))))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_T, "r9_har_rmse.csv"), index=False)
    print(t.round(4).to_string(index=False))
    return t


def tex():
    """Appendix table: Sharpe ratio and fee against the level-calibrated HAR under each trading rule."""
    import paper_tables_final as T
    t = pd.read_csv(os.path.join(OUT_T, "r9_trading.csv"))
    rules = list(dict.fromkeys(t.rule))
    short = {"main (5 bp, w<=2)": "Main", "10 bp": "10 bp", "no leverage (w<=1)": "$w\\le1$",
             "daily rebalancing": "Daily", "drift-aware turnover": "Drift", "10 bp, w<=1, daily": "All three"}
    models = [("P", "har"), ("PMSE", "harx"), ("PMSE", "hars"), ("PM", "lgbmsl-pool"), ("PMC", "hl_gattn-pool"),
              ("PMSE", "hq_meta-pool"), ("PMSE", "transformer-pool"),
              ("PMSE", "combo:hq_meta-pool"), ("PM", "combo:hl_gattn-pool"), ("PM", "combo:lgbmsl-pool")]
    L = []
    for gs, m in models:
        name = ("HAR + " + T.label(gs, m[6:])) if m.startswith("combo:") else T.label(gs, m)
        name = name.replace(", pooled)", ")").replace(" (pooled)", "")
        cells = []
        for r in rules:
            q = t[(t.rule == r) & (t.groupset == gs) & (t.model == m)].iloc[0]
            if (gs, m) == ("P", "har"):
                cells.append(f"{q.sharpe:.2f} & --")
            else:
                star = "$^{*}$" if (q.fee_lo > 0 or q.fee_hi < 0) else ""
                cells.append(f"{q.sharpe:.2f} & {q.fee:+.0f}{star}")
        if m.startswith("combo:") and not L[-1].startswith("\\midrule") and "HAR +" not in L[-1]:
            L.append("\\midrule")
        L.append(name + " & " + " & ".join(cells) + " \\\\")
    tov = " & ".join(f"\\multicolumn{{2}}{{c}}{{{t[(t.rule == r)].turnover.min():.0f}--{t[(t.rule == r)].turnover.max():.0f}}}" for r in rules)
    head = " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{short[r]}}}" for r in rules) + " \\\\\n"
    cm = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(rules))) + "\n"
    sub = "Model, inputs & " + " & ".join("SR & Fee" for _ in rules) + " \\\\\n"
    with open(os.path.join(OUT_T, "r9_trading.tex"), "w") as fh:
        fh.write("% generated by experiments/paper_round9_checks.py tex (from r9_trading.csv)\n"
                 "\\begin{tabular}{l" + "rr" * len(rules) + "}\n\\toprule\n" + head + cm + sub + "\\midrule\n"
                 + "\n".join(L) + "\n\\midrule\nTurnover (all models) & " + tov + " \\\\\n\\bottomrule\n\\end{tabular}\n")
    print(open(os.path.join(OUT_T, "r9_trading.tex")).read())


LGBM_GRID = [dict(num_leaves=nl, min_child_samples=mc) for nl in (15, 31, 63, 127) for mc in (50, 100, 400)]
LGBM_TUNED_DIR = "results/r9_local/final_lgbmtuned_{f}_r9"


def lgbm_tune(freqs=FREQS):
    """The lagged pooled LightGBM-S of paper_round8_checks.lgbm_save (seed 0) refitted for every point of a
    12-point grid over num_leaves and min_child_samples (learning rate 0.03 with early stopping on the
    validation block, as before). The grid point with the lowest validation MSE, summed over coins and
    folds, is selected per frequency; its forecasts are saved like the other runs (model lgbmst-pool)."""
    import json
    import time
    import lightgbm  # noqa: F401
    import src.paper.baselines as B
    from paper_round7_checks import _cfg, _frames
    rows = []
    for f in freqs:
        groups = _cfg(f"lgbm_{f}")["groupsets"]["PM"]
        fr, win = _frames(f)
        dss = list(fr)
        Xs = [fr[ds][0].X(groups) for ds in dss]
        Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in B.HAR_LAGS[f][1:]]) for X in Xs]
        nf = len(fr[dss[0]][1])
        store = {}
        for gi, g in enumerate(LGBM_GRID):
            base = dict(B.LGBM_PARAMS)
            B.LGBM_PARAMS.update(g)
            t0 = time.time()
            vse = 0.0
            res_all = []
            for k in range(nf):
                res = B.predict_baseline_pooled("lgbms", [fr[ds][0].frame for ds in dss], "range",
                                                [fr[ds][1][k] for ds in dss], win, f, Xl, seed=0)
                for ds, r in zip(dss, res):
                    y = fr[ds][0].frame["y_range"].to_numpy(float)
                    vse += float(np.mean((r["val_pred"] - y[r["val_rows"]]) ** 2))
                res_all.append(res)
            B.LGBM_PARAMS.clear()
            B.LGBM_PARAMS.update(base)
            store[gi] = res_all
            rows.append(dict(freq=f, grid=gi, **g, val_mse_sum=vse))
            print(f, g, f"val {vse:.4f}", f"{time.time() - t0:.0f}s", flush=True)
        best = min((r for r in rows if r["freq"] == f), key=lambda r: r["val_mse_sum"])["grid"]
        out = os.path.join(LGBM_TUNED_DIR.format(f=f), "preds")
        os.makedirs(out, exist_ok=True)
        json.dump({"config": {"name": f"final_lgbmtuned_{f}", "note": "paper_round9_checks.lgbm_tune",
                              "params": LGBM_GRID[best]}}, open(os.path.join(os.path.dirname(out), "config.json"), "w"))
        for k, res in enumerate(store[best]):
            for ds, r in zip(dss, res):
                frame = fr[ds][0].frame
                idx = frame.index if frame.index.tz is None else frame.index.tz_convert(None)
                t = idx.as_unit("ns").asi8
                y = frame["y_range"].to_numpy(float)
                np.savez(os.path.join(out, f"{ds}__range__PM__lgbmst-pool__s0__f{k}.npz"),
                         test_time=t[r["test_rows"]], test_y=y[r["test_rows"]].astype("float32"),
                         test_pred=r["test_pred"].astype("float32"), val_time=t[r["val_rows"]],
                         val_y=y[r["val_rows"]].astype("float32"), val_pred=r["val_pred"].astype("float32"),
                         meta=np.array(json.dumps(dict(freq=f, seed=0, fold=k, model="lgbmst-pool", **LGBM_GRID[best]))))
        for r in rows:
            if r["freq"] == f:
                r["selected"] = r["grid"] == best
        pd.DataFrame([r for r in rows if r["freq"] == f]).to_csv(os.path.join(OUT_T, f"r9_lgbm_tune_{f}.csv"), index=False)
    return pd.DataFrame(rows)


def lgbm_tuned_tests(freqs=FREQS):
    """Best calendar network (PMC hl_gattn-pool, seeds 0-2) against the validation-tuned lagged LightGBM-S
    (lgbm_tune, seed 0), and the tuned against the default lagged LightGBM-S (seed 0): mean per-coin MSE
    ratio, coins better, cross-coin DM and the 90% interval of the coin-averaged loss ratio, as in
    paper_round8_checks.bench_tests."""
    from paper_review_checks import cross_losses
    from paper_round7_checks import CAL_NET
    from paper_round8_checks import LGBM_LAG, _ratio_se, _tost_p
    from src.paper.evaluate import dm_cross
    tuned = ("PM", "lgbmst-pool")
    rows = []
    for f in freqs:
        names = [f"final_calendar_pool_{f}", f"final_lgbmlags_{f}", f"final_lgbmtuned_{f}"]
        df = _load(names, lambda g, m: (g, m) in {CAL_NET, LGBM_LAG, tuned})
        df = df[~((df.groupset == CAL_NET[0]) & (df.model == CAL_NET[1]) & (df.seed > 2))]
        df = df[~((df.model == LGBM_LAG[1]) & (df.seed > 0))]
        for a, b in ((CAL_NET, tuned), (tuned, LGBM_LAG), (CAL_NET, LGBM_LAG)):
            xx = df[[(g, m) in (a, b) for g, m in zip(df.groupset, df.model)]]
            if xx.groupby(["groupset", "model"]).ngroups < 2:
                continue
            per = np.array([np.mean((A - y) ** 2) / np.mean((Bp - y) ** 2) for y, A, Bp in _aligned(xx, a, b).values()])
            c = dm_cross(xx.assign(task="range"), a, b)
            r, _, se = _ratio_se(*cross_losses(xx, a, b))
            rows.append(dict(freq=f, a=f"{a[1]} {a[0]}", b=f"{b[1]} {b[0]}", mean_ratio=per.mean(),
                             better=int((per < 1).sum()), n=len(per), dm=c["dm_stat"], p=c["p"], pooled_ratio=r,
                             lo=r - 1.645 * se, hi=r + 1.645 * se, p_tost=_tost_p(r, se)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_T, "r9_lgbm_tuned_tests.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    return t


CAL_LR_RUNS = {f: {1e-3: f"final_calendar_pool_{f}", 3e-3: f"final_lr3e3_calendar_pool_{f}",
                   1e-2: f"final_lr1e2_calendar_pool_{f}"} for f in ("4h", "1h")}


def lr_calendar(freqs=("4h", "1h")):
    """Learning-rate check of the pooled calendar networks (PMC; H-LSTM with group attention and
    H-Transformer with concatenation; seeds 0-2) at 1e-3 (main), 3e-3 and 1e-2 (Colab r9). Per model
    the rate with the lowest validation MSE (mean over coins of the ratio to HAR) is selected; the
    selected network is compared with the lagged pooled LightGBM-S (seed 0) and HAR-S x on the test
    period (mean per-coin ratio, coins better, cross-coin DM)."""
    from paper_round7_checks import _load as load
    from paper_round8_checks import HARSX, LGBM_LAG
    from src.paper.evaluate import dm_cross
    rows, tests = [], []
    for f in freqs:
        har = load([f"final_single_{f}"], lambda g, m: g == "P" and m == "har", splits=("test", "val"))
        hm = {sp: g.groupby("dataset").apply(lambda x: np.mean((x.pred - x.y) ** 2)) for sp, g in har.groupby("split")}
        sel = {}
        for mod in ("hl_gattn-pool", "ht_concat-pool"):
            best = None
            for lrate, name in CAL_LR_RUNS[f].items():
                try:
                    df = load([name], lambda g, m: g == "PMC" and m == mod, splits=("test", "val"), seeds=[0, 1, 2])
                except ValueError:                          # run not (yet) available
                    continue
                e = df.groupby(["split", "dataset", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
                r = {sp: (g.groupby("dataset").apply(lambda x: np.mean((x.pred - x.y) ** 2)) / hm[sp]).mean()
                     for sp, g in e.groupby("split")}
                rows.append(dict(freq=f, model=mod, lr=lrate, val_vs_har=r["val"], test_vs_har=r["test"]))
                if best is None or r["val"] < best[1]:
                    best = (lrate, r["val"], name)
            sel[mod] = best
        bench = load([f"final_lgbmlags_{f}", f"final_harsx_{f}"], lambda g, m: (g, m) in (LGBM_LAG, HARSX))
        bench = bench[~((bench.model == LGBM_LAG[1]) & (bench.seed > 0))]
        for mod, (lrate, _, name) in sel.items():
            net = load([name], lambda g, m: g == "PMC" and m == mod, seeds=[0, 1, 2])
            for b in (LGBM_LAG, HARSX):
                xx = pd.concat([net, bench[(bench.groupset == b[0]) & (bench.model == b[1])]])
                a = ("PMC", mod)
                per = np.array([np.mean((A - y) ** 2) / np.mean((Bp - y) ** 2) for y, A, Bp in _aligned(xx, a, b).values()])
                c = dm_cross(xx.assign(task="range"), a, b)
                tests.append(dict(freq=f, model=mod, lr_selected=lrate, bench=f"{b[1]} {b[0]}", mean_ratio=per.mean(),
                                  better=int((per < 1).sum()), n=len(per), dm=c["dm_stat"], p=c["p"]))
    t1, t2 = pd.DataFrame(rows), pd.DataFrame(tests)
    t1.to_csv(os.path.join(OUT_T, "r9_lr_calendar.csv"), index=False)
    t2.to_csv(os.path.join(OUT_T, "r9_lr_calendar_tests.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t1.round(4).to_string(index=False))
    print(t2.round(4).to_string(index=False))
    return t1, t2


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["clark_west"]:
        name, _, opt = arg.partition(":")
        globals()[name](*([tuple(opt.split(","))] if opt else []))
