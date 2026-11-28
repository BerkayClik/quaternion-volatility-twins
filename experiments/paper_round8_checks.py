"""
Checks for internal review round 8 (October 2026, external-style review of the round-7 manuscript),
from saved forecasts and raw data only; check 1 refits the lagged pooled LightGBM-S of round 7 and
saves its forecasts in the runner's npz format so that every evaluation tool can read them.

    .venv/bin/python -W ignore experiments/paper_round8_checks.py [name ...]
    .venv/bin/python -W ignore experiments/paper_round8_checks.py lgbm_save:4h,1d seeds:0,1,2

1. lgbm_save: pooled LightGBM-S on PM inputs plus HAR-horizon averages of its eight inputs
   (paper_round7_checks.lgbm_lags), seeds 0-2, saved as
   results/r8_local/final_lgbmlags_<freq>_r8/preds/<ds>__range__PM__lgbmsl-pool__s<k>__f<j>.npz
   (test and validation rows, as the runner writes them).
Outputs of the later checks: docs/paper/manuscript/tables/r8_<name>.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from paper_round7_checks import _cfg, _frames  # noqa: E402
from paper_tables_final import OUT_T  # noqa: E402

FREQS = ("4h", "1h", "1d")
LGBM_LAG_DIR = "results/r8_local/final_lgbmlags_{f}_r8"
LGBM_LAG = ("PM", "lgbmsl-pool")


# ============================================================================= 1. lgbm_save
def lgbm_save(freqs=FREQS, seeds=(0, 1, 2)):
    import lightgbm  # noqa: F401  (load its OpenMP runtime first, as the runner does)
    from src.paper.baselines import HAR_LAGS, predict_baseline_pooled
    for f in freqs:
        out = os.path.join(LGBM_LAG_DIR.format(f=f), "preds")
        os.makedirs(out, exist_ok=True)
        json.dump({"config": {"name": f"final_lgbmlags_{f}", "note": "paper_round8_checks.lgbm_save"}},
                  open(os.path.join(os.path.dirname(out), "config.json"), "w"), indent=1)
        groups = _cfg(f"lgbm_{f}")["groupsets"]["PM"]
        fr, win = _frames(f)
        dss = list(fr)
        Xs = [fr[ds][0].X(groups) for ds in dss]
        # HAR-style averages of every input feature over the longer HAR lags (current value = lag 1, already in X)
        Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS[f][1:]]) for X in Xs]
        for seed in seeds:
            for k in range(len(fr[dss[0]][1])):
                paths = [os.path.join(out, f"{ds}__range__{LGBM_LAG[0]}__{LGBM_LAG[1]}__s{seed}__f{k}.npz") for ds in dss]
                if all(os.path.exists(p) for p in paths):
                    continue
                t0 = time.time()
                res = predict_baseline_pooled("lgbms", [fr[ds][0].frame for ds in dss], "range",
                                              [fr[ds][1][k] for ds in dss], win, f, Xl, seed=seed)
                for ds, r, p in zip(dss, res, paths):
                    frame = fr[ds][0].frame
                    idx = frame.index if frame.index.tz is None else frame.index.tz_convert(None)
                    t = idx.as_unit("ns").asi8                         # epoch ns, as the runner saves
                    y = frame["y_range"].to_numpy(float)
                    np.savez(p, test_time=t[r["test_rows"]], test_y=y[r["test_rows"]].astype("float32"),
                             test_pred=r["test_pred"].astype("float32"), val_time=t[r["val_rows"]],
                             val_y=y[r["val_rows"]].astype("float32"), val_pred=r["val_pred"].astype("float32"),
                             meta=np.array(json.dumps(dict(freq=f, seed=seed, fold=k, model="lgbmsl-pool"))))
                print(f, "seed", seed, "fold", k, f"{time.time() - t0:.0f}s", flush=True)


HARSX_DIR = "results/r8_local/final_harsx_{f}_r8"
HARSX = ("P", "hars_x")


# ============================================================================= 1b. harsx_save
def harsx_save(freqs=("4h", "1h")):
    """HAR-S with time-of-day x day-of-week interaction dummies (paper_round7_checks.hars_inter),
    single-coin OLS on each training block; test and validation forecasts saved in the runner format."""
    from paper_round7_checks import _interactions
    from src.paper.baselines import HAR_LAGS, _har_design, _ols
    for f in freqs:
        out = os.path.join(HARSX_DIR.format(f=f), "preds")
        os.makedirs(out, exist_ok=True)
        json.dump({"config": {"name": f"final_harsx_{f}", "note": "paper_round8_checks.harsx_save"}},
                  open(os.path.join(os.path.dirname(out), "config.json"), "w"), indent=1)
        fr, win = _frames(f)
        for ds, (d, folds) in fr.items():
            y = d.frame["y_range"].to_numpy(float)
            D = np.column_stack([_har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[f]),
                                 _interactions(d.frame.index, d.step)])
            idx = d.frame.index if d.frame.index.tz is None else d.frame.index.tz_convert(None)
            t = idx.as_unit("ns").asi8
            for k, fo in enumerate(folds):
                ok = fo["train"][~np.isnan(D[fo["train"]]).any(1)]
                b = _ols(D[ok], y[ok])
                te, va = (r[r >= win - 1] for r in (fo["test"], fo["val"]))
                np.savez(os.path.join(out, f"{ds}__range__{HARSX[0]}__{HARSX[1]}__s0__f{k}.npz"),
                         test_time=t[te], test_y=y[te].astype("float32"),
                         test_pred=(b[0] + D[te] @ b[1:]).astype("float32"), val_time=t[va],
                         val_y=y[va].astype("float32"), val_pred=(b[0] + D[va] @ b[1:]).astype("float32"),
                         meta=np.array(json.dumps(dict(freq=f, fold=k, model="hars_x"))))
        print(f, "saved", flush=True)


# ============================================================================= 1c. bench_tests
def bench_tests():
    """Best calendar network (pooled H-LSTM with group attention, PMC, seeds 0-2) against the lagged
    pooled LightGBM-S (seed 0 and the 3-seed average) and HAR-S x: mean of per-coin MSE ratios, coins
    better, cross-coin DM (as Table 7) and the +-1% equivalence interval of the coin-averaged loss
    ratio (exact delta-method se)."""
    from paper_review_checks import cross_losses
    from paper_round7_checks import CAL_NET, _load
    from src.paper.evaluate import dm_cross
    rows = []
    for f in FREQS:
        names = [f"final_single_{f}", f"final_calendar_pool_{f}", f"final_lgbm_{f}", f"final_lgbmlags_{f}",
                 f"final_harsx_{f}", f"final_har_season_{f}"]
        keys = {("P", "har"), CAL_NET, LGBM_LAG, HARSX, ("PM", "lgbms-pool"), ("PMSE", "hars")}
        df = _load(names, lambda g, m: (g, m) in keys)
        df = df[~((df.groupset == CAL_NET[0]) & (df.model == CAL_NET[1]) & (df.seed > 2))]
        variants = {"lgbm_lag_s0": df[~((df.model == LGBM_LAG[1]) & (df.seed > 0))], "lgbm_lag_3s": df}
        for vname, x in variants.items():
            for a, b in ((LGBM_LAG, ("P", "har")), (CAL_NET, LGBM_LAG), (HARSX, ("P", "har")), (CAL_NET, HARSX),
                         (LGBM_LAG, ("PM", "lgbms-pool")), (CAL_NET, ("PMSE", "hars"))):
                if vname == "lgbm_lag_3s" and LGBM_LAG not in (a, b):
                    continue
                xx = x[[(g, m) in (a, b) for g, m in zip(x.groupset, x.model)]]
                if xx.groupby(["groupset", "model"]).ngroups < 2:
                    continue
                e = xx.groupby(["dataset", "groupset", "model", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
                per = []
                for ds, g in e.groupby("dataset"):
                    A = g[(g.groupset == a[0]) & (g.model == a[1])].set_index("time")
                    B = g[(g.groupset == b[0]) & (g.model == b[1])].set_index("time")
                    t = A.index.intersection(B.index)
                    per.append(np.mean((A.loc[t, "pred"] - A.loc[t, "y"]) ** 2) / np.mean((B.loc[t, "pred"] - B.loc[t, "y"]) ** 2))
                per = np.array(per)
                c = dm_cross(xx.assign(task="range"), a, b)
                r, _, se = _ratio_se(*cross_losses(xx, a, b))
                rows.append(dict(freq=f, variant=vname, a=f"{a[1]} {a[0]}", b=f"{b[1]} {b[0]}", mean_ratio=per.mean(),
                                 better=int((per < 1).sum()), n=len(per), dm=c["dm_stat"], p=c["p"], pooled_ratio=r,
                                 lo=r - 1.645 * se, hi=r + 1.645 * se, p_tost=_tost_p(r, se),
                                 equivalent=bool(r - 1.645 * se > 0.99 and r + 1.645 * se < 1.01)))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "r8_bench_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    return t


# ============================================================================= 1d. mcs_bench
def mcs_bench():
    """The 90% MCS of Appendix E.2 (review_checks.main_mcs_calendar settings, seeds 0-2) with the
    lagged pooled LightGBM-S (seeds 0-2) and HAR-S x added."""
    from arch.bootstrap import MCS
    from paper_review_checks import RUNS, ensemble, load_both
    rows = []
    for f in FREQS:
        names = RUNS[f] + [f"final_calendar_pool_{f}", f"final_calendar2_pool_{f}", f"final_lgbmlags_{f}", f"final_harsx_{f}"]
        E = ensemble(load_both(names), seeds=[0, 1, 2])
        E = E[(E.split == "test") & ~E.model.isin(["mean"])]
        E["se"] = (E.pred - E.y) ** 2
        L = E.pivot_table(index="time", columns=["groupset", "model"], values="se", aggfunc="mean").dropna(axis=1)
        drop = {"persistence", "ridge", "garch", "gjr"}
        L = L.loc[:, [c for c in L.columns if c[1] not in drop and not (c[1] == "har" and c[0] != "P")]]
        L = L.T.drop_duplicates().T
        mcs = MCS(L.to_numpy(), size=0.10, reps=2000, block_size=max(2, int(len(L) ** (1 / 3))), seed=0)
        mcs.compute()
        names_in = [f"{L.columns[i][1]} {L.columns[i][0]}" for i in mcs.included]
        pv = mcs.pvalues.squeeze()
        rows.append(dict(freq=f, n_models=L.shape[1], n_mcs=len(names_in), members="; ".join(names_in),
                         p_lgbm_lag=float(pv.get(list(L.columns).index(LGBM_LAG), np.nan)) if LGBM_LAG in L.columns else np.nan,
                         p_harsx=float(pv.get(list(L.columns).index(HARSX), np.nan)) if HARSX in L.columns else np.nan,
                         p_cal_net=float(pv.get(list(L.columns).index(("PMC", "hl_gattn-pool")), np.nan))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT_T / "r8_mcs_bench.csv", index=False)


# ============================================================================= 3. econ_cal
def _recal(V, Vv, freq, target=0.40, w_max=2.0, cost=0.0005):
    """Every model's variance forecast x k (one k per model, coin and fold), k set so that the targeted
    strategy's realised volatility on that fold's validation block equals the target (the rule of the
    level-calibrated HAR, paper_round7_checks._recal_har, applied to all models). Validation data only.
    Returns the recalibrated test rows (model name + '|lvl') and the factors."""
    from scipy import optimize
    from src.paper.economic import BARS_PER_YEAR
    n = BARS_PER_YEAR[freq]
    tgt = target / np.sqrt(n)
    vv = {k: g.sort_values("time") for k, g in Vv.groupby(["groupset", "model", "dataset", "fold"])}
    out, ks = [], []
    for (gs, m, ds, fold), h in V.groupby(["groupset", "model", "dataset", "fold"]):
        v = vv[(gs, m, ds, fold)]
        simple, var = np.expm1(v.r.to_numpy()), v["var"].to_numpy()

        def gap(logk):
            w = np.minimum(w_max, tgt / np.sqrt(np.exp(logk) * var))
            rp = w * simple - cost * np.abs(np.diff(w, prepend=w[0]))
            return np.std(rp) * np.sqrt(n) - target
        k = np.exp(optimize.brentq(gap, -8, 8))
        ks.append(dict(groupset=gs, model=m, dataset=ds, fold=fold, k=k))
        out.append(h.assign(model=m + "|lvl", var=k * h["var"].to_numpy()))
    return pd.concat(out, ignore_index=True), pd.DataFrame(ks)


def econ_cal(reps=1000, seed=0):
    """Table 13 (4-hourly volatility targeting) with every model, combination and the lagged LightGBM-S
    (and its and the calendar network's 50/50 combinations with HAR) recalibrated by the strategy-
    volatility rule of the level-calibrated HAR. Fees (gamma = 10, 5 bp) against plain HAR and against the
    level-calibrated HAR, with moving-block bootstrap 90% intervals (as paper_round7_checks.econ_level)."""
    import paper_tables_final as T
    from paper_round7_checks import _rv30
    from src.paper.economic import BARS_PER_YEAR, fko_fee, load_preds, variance_forecasts, vol_target
    freq = "4h"
    n = BARS_PER_YEAR[freq]
    econ = T.ECON + [LGBM_LAG]
    T.COMBOS = T.COMBOS + [("PMC", "hl_gattn-pool"), LGBM_LAG]          # _add_combos reads these globals
    dirs = [T.run_dir(f"final_{x}_{freq}") for x in ("single", "pool", "harx", "har_season", "lgbm", "calendar_pool")]
    dirs.append(LGBM_LAG_DIR.format(f=freq))
    keep = lambda d: d[[(g, m) in econ for g, m in zip(d.groupset, d.model)]]
    test = pd.concat([keep(load_preds(d)) for d in dirs if d])
    val = pd.concat([keep(load_preds(d, split="val")) for d in dirs if d])
    rt, rv = T._harrv_preds(test, val, freq)
    test, val = pd.concat([test, rt], ignore_index=True), pd.concat([val, rv], ignore_index=True)
    Vv = variance_forecasts(val, val)
    V = T._add_combos(variance_forecasts(test, val), Vv)
    Vv = T._add_combos(Vv, Vv)
    L, ks = _recal(V, Vv, freq)
    V = pd.concat([V, L], ignore_index=True)
    print("k by model (median):", ks.groupby("model").k.median().round(3).to_dict(), flush=True)
    res, series = vol_target(V, freq)
    R, _ = _rv30(V, series, n)
    T_ = len(series[next(iter(series))])
    coins = sorted({k[0] for k in series})
    rng = np.random.default_rng(seed)
    Lb = max(2, int(T_ ** (1 / 3)))
    nb = -(-T_ // Lb)
    draws = [(rng.integers(0, T_ - Lb + 1, nb)[:, None] + np.arange(Lb)).ravel()[:T_] for _ in range(reps)]
    fee = lambda m, b, ix=slice(None): np.mean([fko_fee(series[(c,) + m][ix], series[(c,) + b][ix], 10) * n * 1e4
                                               for c in coins])
    base = [k for k in dict.fromkeys(zip(V.groupset, V.model)) if not k[1].endswith("|lvl")]
    rows = []
    for gs, m in base:
        rec = dict(groupset=gs, model=m)
        for tag, mm in (("", m), ("lvl_", m + "|lvl")):
            g = res[(res.groupset == gs) & (res.model == mm)]
            Rm = R[(gs, mm)]
            rec.update({f"{tag}vol_err": g.vol_err.mean(), f"{tag}real_vol": g.ann_vol.mean(),
                        f"{tag}rmsd30": float(np.sqrt(np.nanmean((Rm.to_numpy() - 0.40) ** 2))),
                        f"{tag}sharpe": g.sharpe.mean(), f"{tag}mdd": g.mdd.mean(), f"{tag}turnover": g.turnover.mean()})
            for btag, b in (("har", ("P", "har")), ("harlvl", ("P", "har|lvl"))):
                if (gs, mm) == b:
                    continue
                bs = np.array([fee((gs, mm), b, ix) for ix in draws])
                rec.update({f"{tag}fee_{btag}": fee((gs, mm), b), f"{tag}fee_{btag}_lo": np.percentile(bs, 5),
                            f"{tag}fee_{btag}_hi": np.percentile(bs, 95),
                            f"{tag}fee_{btag}_pos": int(sum(fko_fee(series[(c, gs, mm)], series[(c,) + b], 10) > 0
                                                            for c in coins))})
        rows.append(rec)
        print(gs, m, flush=True)
    t = pd.DataFrame(rows)
    # sanity: the level-calibrated HAR and the uncalibrated fees must reproduce i7_econ_level.csv
    e = pd.read_csv(OUT_T / "i7_econ_level.csv").set_index(["groupset", "model"])
    h = t.set_index(["groupset", "model"])
    common = [k for k in h.index if k in e.index and pd.notna(e.loc[k, "fee_har"])]
    print("check vs i7_econ_level: max |fee_har diff|",
          f"{max(abs(h.loc[k, 'fee_har'] - e.loc[k, 'fee_har']) for k in common):.3g}",
          "| HAR|lvl vol_err", round(h.loc[("P", "har"), "lvl_vol_err"], 4), "vs", round(e.loc[("P", "har_lvl"), "vol_err"], 4))
    t.to_csv(OUT_T / "r8_econ_cal.csv", index=False)
    pd.set_option("display.width", 250)
    print(t[["groupset", "model", "vol_err", "rmsd30", "fee_har", "fee_harlvl", "lvl_vol_err", "lvl_rmsd30", "lvl_sharpe",
             "lvl_fee_har", "lvl_fee_harlvl", "lvl_fee_harlvl_lo", "lvl_fee_harlvl_hi", "lvl_fee_harlvl_pos"]]
          .round(3).to_string(index=False))
    return t


# ============================================================================= 4. pt_hac
def _pt_hac(y, p):
    """Serial-correlation-robust directional test (Pesaran and Timmermann 2009, regression form): OLS of
    1{y > 0} on 1{p > 0} with a Newey-West (Bartlett, floor(n^(1/3)) lags) standard error on the slope;
    one-sided p (positive dependence). Zero returns dropped, as in evaluate.pesaran_timmermann."""
    from scipy import stats
    from paper_round7_checks import _nw_lrv
    m = y != 0
    a, b = (y[m] > 0).astype(float), (p[m] > 0).astype(float)
    n = len(a)
    if n < 10 or b.std() == 0:
        return np.nan
    bc = b - b.mean()
    beta = (bc @ (a - a.mean())) / (bc @ bc)
    u = (a - a.mean()) - beta * bc
    lag = int(np.floor(n ** (1 / 3)))
    se = np.sqrt(_nw_lrv(bc * u, lag) * n) / (bc @ bc)
    return float(stats.norm.sf(beta / se))


def pt_hac():
    """Appendix (returns task): the PT test count (seed-mean p < 0.05, as paper_tables_final.returns_table)
    next to the HAC regression version."""
    from paper_tables_final import run
    from src.paper.evaluate import pesaran_timmermann
    df = pd.concat([x for x in (run("final_return_4h"), run("final_return_pool_4h")) if x is not None])
    df = df[(df.task == "return") & (df.model != "persistence") & ~((df.model == "mean") & (df.groupset != "P"))]
    rows = []
    for key, g in df.groupby(["dataset", "groupset", "model", "seed"]):
        g = g.sort_values("time")
        y, p = g.y.to_numpy(), g.pred.to_numpy()
        rows.append(dict(zip(["dataset", "groupset", "model", "seed"], key), pt=pesaran_timmermann(y, p), pt_hac=_pt_hac(y, p)))
    s = pd.DataFrame(rows).groupby(["dataset", "groupset", "model"])[["pt", "pt_hac"]].mean().reset_index()
    out = dict(cases=int(s.pt.notna().sum()), pt_sig=int((s.pt < .05).sum()), pt_hac_sig=int((s.pt_hac < .05).sum()),
               both=int(((s.pt < .05) & (s.pt_hac < .05)).sum()))
    print(out)
    pd.DataFrame([out]).to_csv(OUT_T / "r8_pt_hac.csv", index=False)
    return out


# ============================================================================= 5. ffill_share
def ffill_share():
    """Share of training, validation and test rows whose LunarCrush values are carried forward (no
    observation in that bar; at most one day, data.FFILL_LIMIT) and, for 4-hourly bars, whose values are
    built from 2 or 3 of the 4 hourly observations (event counts rescaled by 4/n; scripts/download_lunarcrush_paper.to_4h).
    The sample start of every coin (after its last longer outage) lies before its first validation block."""
    from src.paper.data import load_raw
    rows = []
    for f in FREQS:
        fr, _ = _frames(f)
        for ds, (d, folds) in fr.items():
            coin = ds.split("_")[0]
            _, lunar = load_raw(coin, f)
            lunar.index = pd.to_datetime(lunar.index, utc=True)
            idx = d.frame.index if d.frame.index.tz is not None else d.frame.index.tz_localize("UTC")
            used = ["contributors_active", "contributors_created", "posts_active", "posts_created", "sentiment",
                    "galaxy_score", "social_dominance", "interactions"]          # the columns data.build_features uses
            obs = lunar[used].reindex(idx).notna().all(axis=1).to_numpy()
            partial = np.zeros(len(idx), bool)
            if f == "4h":
                h = pd.read_csv(f"data/lunarcrush/{coin}_hour.csv", index_col=0, parse_dates=True)
                h.index = pd.to_datetime(h.index, utc=True)
                n = h[used].notna().all(axis=1).resample("4h").sum().reindex(idx).fillna(0).to_numpy()
                partial = obs & (n < 4)
            val0 = idx[folds[0]["val"][0]]
            for split, sel in (("train (last fold)", folds[-1]["train"]),
                               ("val", np.concatenate([k["val"] for k in folds])),
                               ("test", np.concatenate([k["test"] for k in folds]))):
                sel = np.unique(sel)
                rows.append(dict(freq=f, dataset=ds, split=split, bars=len(sel), ffilled=int((~obs[sel]).sum()),
                                 partial=int(partial[sel].sum()), start=str(idx[0]), first_val=str(val0),
                                 start_before_val=bool(idx[0] < val0)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "r8_ffill_share.csv", index=False)
    s = t.groupby(["freq", "split"], sort=False)[["bars", "ffilled", "partial"]].sum()
    s["ffill_pct"], s["partial_pct"] = 100 * s.ffilled / s.bars, 100 * s.partial / s.bars
    mx = t.assign(p=100 * t.ffilled / t.bars).groupby(["freq", "split"], sort=False).p.max().rename("ffill_pct_max_coin")
    print(s.join(mx).round(3).to_string())
    print("all starts before first validation block:", bool(t.start_before_val.all()))
    return t


# ============================================================================= 6. harfam_trainval
def harfam_trainval():
    """HAR family fitted on the training block (as in the paper) against a refit on training + validation
    blocks: test MSE ratio (refit / paper) per coin, for HAR, HAR-S, HAR-XS (PMSE) and HAR-S x (4h, 1h).
    OLS is invariant to the affine train-fold scaling of the regressors, so raw features are used."""
    from paper_round7_checks import _interactions
    from src.paper.baselines import HAR_LAGS, _design, _har_design, _ols
    rows = []
    for f in FREQS:
        fr, win = _frames(f)
        groups = _cfg(f"harx_{f}")["groupsets"]["PMSE"] if os.path.exists(f"configs/paper/final/harx_{f}.yaml") else None
        for ds, (d, folds) in fr.items():
            y = d.frame["y_range"].to_numpy(float)
            X = d.X(groups)
            designs = {name: _design(name, d.frame, f, X) for name in ("har", "hars", "harxs")}
            if f != "1d":
                designs["hars_x"] = np.column_stack([_har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[f]),
                                                     _interactions(d.frame.index, d.step)])
            for name, D in designs.items():
                se = {"train": 0.0, "train+val": 0.0}
                for fo in folds:
                    te = fo["test"][fo["test"] >= win - 1]
                    for tag, tr in (("train", fo["train"]), ("train+val", np.r_[fo["train"], fo["val"]])):
                        ok = tr[~np.isnan(D[tr]).any(1) & np.isfinite(y[tr])]
                        b = _ols(D[ok], y[ok])
                        se[tag] += ((b[0] + D[te] @ b[1:] - y[te]) ** 2).sum()
                rows.append(dict(freq=f, dataset=ds, model=name, ratio=se["train+val"] / se["train"]))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    out = t.groupby(["freq", "model"], sort=False).ratio.agg(mean="mean", min="min", max="max",
                                                             lower=lambda x: int((x < 1).sum())).reset_index()
    out.to_csv(OUT_T / "r8_harfam_trainval.csv", index=False)
    print(out.round(4).to_string(index=False))
    return out


# ============================================================================= 8. prmc
def prmc(root=None, freqs=FREQS):
    """Calendar networks with the current log range in the real part of the price quaternion (PRMC,
    configs/paper/final/prmc_pool_<freq>.yaml, seeds 0-2) against the same networks on PMC (seeds 0-2),
    HAR, the lagged pooled LightGBM-S (seeds 0-2) and HAR-S x: mean of per-coin MSE ratios, coins better,
    cross-coin DM and the 90% interval of the coin-averaged loss ratio. `root`: directory holding the
    final_prmc_pool_<freq>_* runs (default: the results tree)."""
    import glob as _glob
    from paper_review_checks import cross_losses
    from paper_round7_checks import _load
    from src.paper.evaluate import _to_time, dm_cross
    rows = []
    for f in freqs:
        d = (_glob.glob(f"{root}/final_prmc_pool_{f}_*") if root else [None])[0]
        names = [f"final_single_{f}", f"final_calendar_pool_{f}", f"final_lgbmlags_{f}", f"final_harsx_{f}"]
        if d is None:
            names.append(f"final_prmc_pool_{f}")
        keys = {("P", "har"), ("PMC", "hl_gattn-pool"), ("PMC", "ht_concat-pool"), LGBM_LAG, HARSX}
        df = _load(names, lambda g, m: (g, m) in keys or g == "PRMC")
        if d is not None:
            parts = []
            for path in _glob.glob(os.path.join(d, "preds", "*__range__*.npz")):
                ds, task, gs, model, seed, fold = os.path.basename(path)[:-4].split("__")
                z = np.load(path)
                parts.append(pd.DataFrame({"dataset": ds, "task": task, "groupset": gs, "model": model,
                                           "seed": int(seed[1:]), "fold": int(fold[1:]), "split": "test",
                                           "time": _to_time(z["test_time"]), "y": z["test_y"].astype(float),
                                           "pred": z["test_pred"].astype(float)}))
            df = pd.concat([df] + parts, ignore_index=True)
        df = df[(df.seed <= 2)]
        pairs = []
        for m in ("hl_gattn-pool", "ht_concat-pool"):
            a = ("PRMC", m)
            pairs += [(a, ("P", "har")), (a, ("PMC", m)), (a, LGBM_LAG), (a, HARSX)]
        for a, b in pairs:
            xx = df[[(g, mm) in (a, b) for g, mm in zip(df.groupset, df.model)]]
            if xx.groupby(["groupset", "model"]).ngroups < 2:
                continue
            e = xx.groupby(["dataset", "groupset", "model", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
            per = []
            for ds, g in e.groupby("dataset"):
                A = g[(g.groupset == a[0]) & (g.model == a[1])].set_index("time")
                B = g[(g.groupset == b[0]) & (g.model == b[1])].set_index("time")
                t = A.index.intersection(B.index)
                per.append(np.mean((A.loc[t, "pred"] - A.loc[t, "y"]) ** 2) / np.mean((B.loc[t, "pred"] - B.loc[t, "y"]) ** 2))
            per = np.array(per)
            c = dm_cross(xx.assign(task="range"), a, b)
            r, _, se = _ratio_se(*cross_losses(xx, a, b))
            rows.append(dict(freq=f, a=f"{a[1]} {a[0]}", b=f"{b[1]} {b[0]}", mean_ratio=per.mean(),
                             better=int((per < 1).sum()), n=len(per), dm=c["dm_stat"], p=c["p"], pooled_ratio=r,
                             lo=r - 1.645 * se, hi=r + 1.645 * se))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "r8_prmc.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    return t


# ============================================================================= 7. tex
NAMES = {"lstm": "LSTM", "qlstm": "QLSTM", "transformer": "Transformer", "qtransformer": "QTransformer",
         "hl_meta": "H-LSTM (meta)", "hq_meta": "\\HQ{} (meta)", "ht_meta": "H-Transformer (meta)",
         "hqt_meta": "H-QTransformer (meta)", "hl_gattn": "H-LSTM (group attn.)", "hq_gattn": "\\HQ{} (group attn.)"}


def tex():
    """LaTeX tables of the round-8 appendix: learning-rate sweep, matched Transformer control, network
    against the stronger benchmarks, and volatility targeting with every forecast level-calibrated."""
    def p_(p):
        return "$<$0.001" if p < 0.001 else f"{p:.3f}"
    # (a) learning rate
    L = []
    for st, title in (("single_1d", "Single-coin, daily"), ("pool_4h", "Pooled, 4-hourly")):
        m = pd.read_csv(OUT_T / f"i7c_lr_models_{st}.csv")
        L.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for model, g in m.groupby("model", sort=False):
            g = g.set_index("lr")
            sel = g.val_vs_har.idxmin()
            cells = [(f"\\textbf{{{g.test_vs_har[lr]:.3f}}}" if lr == sel else f"{g.test_vs_har[lr]:.3f}")
                     for lr in (0.0003, 0.001, 0.003)]
            L.append(f"{NAMES[model.replace('-pool', '')]} & " + " & ".join(cells) + f" & {g.test_vs_har[sel] / g.test_vs_har[0.001]:.3f} \\\\")
    for st in ("single_1d", "pool_4h"):
        tw = pd.read_csv(OUT_T / f"i7c_lr_twins_{st}.csv")
        L.append(f"\\midrule\n\\multicolumn{{5}}{{l}}{{\\emph{{Quaternion / real twin, {'single-coin daily' if st == 'single_1d' else 'pooled 4-hourly'}}}}} \\\\")
        for r in tw.itertuples():
            pair = r.pair.replace(", PMSE", "").replace(" (pooled)", "").replace(", pooled)", ")").replace("H-QLSTM", "\\HQ{}")
            L.append(f"{pair} & {getattr(r, 'lr0_0003', r[3]):.3f} & {r[4]:.3f} & \\textbf{{{r[5]:.3f}}} & -- \\\\")
    (OUT_T / "r8_lr.tex").write_text(
        "% generated by experiments/paper_round8_checks.py tex (from i7c_lr_*.csv)\n\\begin{tabular}{lrrrr}\n\\toprule\n"
        " & \\multicolumn{3}{c}{Learning rate} & \\\\\n\\cmidrule(lr){2-4}\n"
        "Model (PMSE) & $3{\\times}10^{-4}$ & $10^{-3}$ & $3{\\times}10^{-3}$ & Sel./$10^{-3}$ \\\\\n\\midrule\n"
        + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    # (b) matched Transformer control
    mt = pd.read_csv(OUT_T / "review_matched.csv")
    lab = {"qtransformer/transformer_g": "QTransformer / Transformer", "hqt_meta/htg_meta": "H-QTransformer / H-Transformer (meta)",
           "hqt_concat/htg_concat": "H-QTransformer / H-Transformer (concat)"}
    L = []
    for r in mt.itertuples():
        L.append(f"{'4-hourly' if r.freq == '4h' else 'Hourly'} & {lab[r.pair]} & {r.ratio_orig:.3f} & {p_(r.p_orig)} & "
                 f"{r.ratio:.3f} & {p_(r.p)} & {r.better}/{r.n} & [{r.eq_ci_lo:.3f}, {r.eq_ci_hi:.3f}] \\\\")
    (OUT_T / "r8_matched.tex").write_text(
        "% generated by experiments/paper_round8_checks.py tex (from review_matched.csv)\n\\begin{tabular}{llrrrrrr}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{Main grid} & \\multicolumn{4}{c}{Matched real twin} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\n"
        "Freq. & Pair (pooled, PMSE) & Ratio & DM $p$ & Ratio & DM $p$ & Q better & 90\\% CI \\\\\n\\midrule\n"
        + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    # (c) best calendar network against the stronger benchmarks
    bt = pd.read_csv(OUT_T / "r8_bench_tests.csv")
    fn = {"4h": "4-hourly", "1h": "Hourly", "1d": "Daily"}
    rows = [("lgbm_lag_s0", "hl_gattn-pool PMC", "lgbmsl-pool PM", "LightGBM-S (lagged), seed 0"),
            ("lgbm_lag_3s", "hl_gattn-pool PMC", "lgbmsl-pool PM", "LightGBM-S (lagged), 3 seeds"),
            ("lgbm_lag_s0", "hl_gattn-pool PMC", "hars_x P", "HAR-S$\\times$"),
            ("lgbm_lag_s0", "hl_gattn-pool PMC", "hars PMSE", "HAR-S")]
    L = []
    for f in FREQS:
        for v, a, b, nm in rows:
            r = bt[(bt.freq == f) & (bt.variant == v) & (bt.a == a) & (bt.b == b)]
            if not len(r):
                continue
            r = r.iloc[0]
            L.append(f"{fn[f]} & {nm} & {r.mean_ratio:.3f} & {r.better}/{r.n} & {r.dm:.2f} & {p_(r.p)} & "
                     f"[{r.lo:.3f}, {r.hi:.3f}] \\\\")
    pr = pd.read_csv(OUT_T / "r8_prmc.csv")
    pr_rows = [("hl_gattn-pool PRMC", "lgbmsl-pool PM", "PRMC network vs LightGBM-S (lagged), 3 seeds"),
               ("hl_gattn-pool PRMC", "hars_x P", "PRMC network vs HAR-S$\\times$")]
    L.append("\\midrule")
    for f in FREQS:
        for a, b, nm in pr_rows:
            r = pr[(pr.freq == f) & (pr.a == a) & (pr.b == b)]
            if not len(r):
                continue
            r = r.iloc[0]
            L.append(f"{fn[f]} & {nm} & {r.mean_ratio:.3f} & {r.better}/{r.n} & {r.dm:.2f} & {p_(r.p)} & "
                     f"[{r.lo:.3f}, {r.hi:.3f}] \\\\")
    (OUT_T / "r8_bench.tex").write_text(
        "% generated by experiments/paper_round8_checks.py tex (from r8_bench_tests.csv, r8_prmc.csv)\n\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "Freq. & Benchmark $B$ & MSE/$B$ & better & DM & $p$ & 90\\% CI \\\\\n\\midrule\n"
        + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    # (c') compact version for the main text: ratio and DM p per frequency
    cells = {}
    for f in FREQS:
        for key, src, v, a, b in [
                ("hs", bt, "lgbm_lag_s0", "hl_gattn-pool PMC", "hars PMSE"),
                ("hsx", bt, "lgbm_lag_s0", "hl_gattn-pool PMC", "hars_x P"),
                ("lg", bt, "lgbm_lag_3s", "hl_gattn-pool PMC", "lgbmsl-pool PM"),
                ("prlg", pr, None, "hl_gattn-pool PRMC", "lgbmsl-pool PM")]:
            q = src[(src.freq == f) & (src.a == a) & (src.b == b)]
            if v is not None:
                q = q[q.variant == v]
            cells[key, f] = (f"{q.iloc[0].mean_ratio:.3f}", p_(q.iloc[0].p)) if len(q) else ("--", "--")
        # round 11: clipped HAR-XS x (paper_round11_checks.frontier; on daily bars = clipped HAR-XS)
        fr11 = pd.read_csv(OUT_T / "r11_frontier.csv")
        q = fr11[(fr11.freq == f) & (fr11.a == "hl_gattn-pool PMC") & (fr11.b == "harxs_x PMSE")]
        cells["hxsx", f] = (f"{q.iloc[0].mean_ratio:.3f}", p_(q.iloc[0].p)) if len(q) else ("--", "--")
    labels = {"hs": "HAR-S", "hsx": "HAR-S$\\times$", "hxsx": "HAR-XS$\\times$ (clipped)",
              "lg": "LightGBM-S (lagged)", "prlg": "LightGBM-S (lagged), PRMC network"}
    L = [f"{labels[k]} & " + " & ".join(f"{cells[k, f][0]} & {cells[k, f][1]}" for f in ("1h", "4h", "1d")) + " \\\\"
         for k in ("hs", "hsx", "hxsx", "lg", "prlg")]
    (OUT_T / "r8_bench_main.tex").write_text(
        "% generated by experiments/paper_round8_checks.py tex (from r8_bench_tests.csv, r8_prmc.csv)\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Hourly} & \\multicolumn{2}{c}{4-hourly} & \\multicolumn{2}{c}{Daily} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Benchmark $B$ & MSE/$B$ & $p$ & MSE/$B$ & $p$ & MSE/$B$ & $p$ \\\\\n\\midrule\n"
        + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    # (d) volatility targeting, every forecast level-calibrated
    import paper_tables_final as T
    ec = pd.read_csv(OUT_T / "r8_econ_cal.csv")
    order = [("P", "har")] + T.ECON[1:] + T.ECON_R8 + [(g, "combo:" + m) for g, m in T.COMBOS + T.COMBOS_R8 +
                                                     T.COMBOS_BENCH[:2] + T.COMBOS_BENCH_R8]
    L = []
    for gs, m in order:
        r = ec[(ec.groupset == gs) & (ec.model == m)]
        if not len(r):
            continue
        r = r.iloc[0]
        name = ("HAR + " + T.label(gs, m[6:])) if m.startswith("combo:") else T.label(gs, m)
        name = name.replace(", pooled)", ")").replace(" (pooled)", "")
        fee = "--" if (gs, m) == ("P", "har") else f"{r.lvl_fee_harlvl:+.0f} [{r.lvl_fee_harlvl_lo:+.0f}, {r.lvl_fee_harlvl_hi:+.0f}]"
        pos = "--" if (gs, m) == ("P", "har") else f"{int(r.lvl_fee_harlvl_pos)}/15"
        if m.startswith("combo:") and not any(x.startswith("\\midrule") for x in L[-1:]) and "HAR +" not in (L[-1] if L else ""):
            L.append("\\midrule")
        L.append(f"{name} & {r.vol_err:.3f} & {r.lvl_vol_err:.3f} & {r.lvl_rmsd30:.3f} & {r.lvl_sharpe:.2f} & {fee} & {pos} \\\\")
    (OUT_T / "r8_econ_cal.tex").write_text(
        "% generated by experiments/paper_round8_checks.py tex (from r8_econ_cal.csv)\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        " & \\multicolumn{1}{c}{Before} & \\multicolumn{5}{c}{Every forecast level-calibrated} \\\\\n"
        "\\cmidrule(lr){2-2}\\cmidrule(lr){3-7}\n"
        "Model, inputs & Vol.\\ err. & Vol.\\ err. & RMSD$_{30}$ & Sharpe & Fee vs cal.\\ HAR & $>0$ \\\\\n\\midrule\n"
        + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    print("written r8_lr.tex, r8_matched.tex, r8_bench.tex, r8_econ_cal.tex")


# ============================================================================= 2. equiv_exact
def _ratio_se(LA, LB, lag=None):
    """Loss ratio r = mean(LA) / mean(LB) with two delta-method standard errors: at r0 = 1 (the
    paper's construction, NW variance of d = LA - LB) and at the sample ratio (NW variance of
    d - (r - 1) LB, the exact first-order term)."""
    from paper_round7_checks import _nw_lrv
    a, b = np.asarray(LA, float), np.asarray(LB, float)
    n, lb = len(a), b.mean()
    lag = int(np.floor(n ** (1 / 3))) if lag is None else lag
    d = a - b
    r = 1 + d.mean() / lb
    se0 = np.sqrt(_nw_lrv(d, lag) / n) / lb
    se1 = np.sqrt(_nw_lrv(d - (r - 1) * b, lag) / n) / lb
    return r, se0, se1


def _tost_p(r, se, margin=0.01):
    """Two one-sided tests of |r - 1| < margin (normal approximation): p = max of the two one-sided p."""
    from scipy import stats
    return float(max(stats.norm.sf((r - (1 - margin)) / se), stats.norm.sf(((1 + margin) - r) / se)))


def equiv_exact(margin=0.01):
    """The 46 +-1% equivalence tests (42 twins of Fig. 2, 4 nonlinear-fusion pairs) with (a) the paper's
    se (at r0 = 1), (b) the exact delta-method se at the sample ratio, (c) (b) plus the between-seed
    variance of i7_seed_equiv (twins only), each also with Holm across all 46 (42) TOST p-values."""
    from paper_review_checks import cross_losses, equivalence
    from paper_round7_checks import _load
    from paper_tables_final import POOL, TWINS, run
    from src.paper.evaluate import holm
    seq = pd.read_csv(OUT_T / "i7_seed_equiv.csv").set_index(["freq", "setting", "inputs", "pair"])
    rows = []
    for f in FREQS:
        keep = lambda g, m: g in ("P", "PMSE") and m.replace(POOL, "") in {x for t in TWINS for x in t}
        df = _load([f"final_single_{f}", f"final_pool_{f}"], keep)
        for tag, sfx in (("single", ""), ("pooled", POOL)):
            for gs in ("P", "PMSE"):
                for q, rr in TWINS:
                    a, b = (gs, q + sfx), (gs, rr + sfx)
                    x = df[[(g, m) in (a, b) for g, m in zip(df.groupset, df.model)]]
                    if x.model.nunique() < 2:
                        continue
                    r, se0, se1 = _ratio_se(*cross_losses(x, a, b))
                    se_seed = seq.se_seed.get((f, tag, gs, f"{q}/{rr}"), np.nan)
                    rows.append(dict(freq=f, setting=tag, inputs=gs, pair=f"{q}/{rr}", ratio=r, se_r0=se0,
                                     se_exact=se1, se_seed=se_seed))
        print(f, flush=True)
    fnl = run("final_fusion_nl_pool_4h")
    pool = run("final_pool_4h")
    pool = pool[(pool.task == "range") & (pool.seed <= 2) & (pool.groupset == "PMSE")]
    x = pd.concat([pool, fnl[fnl.task == "range"]])
    for enc in ("hq", "hl"):
        for fu in ("phmn", "densen"):
            a, b = ("PMSE", f"{enc}_{fu}-pool"), ("PMSE", f"{enc}_metan-pool")
            r, se0, se1 = _ratio_se(*cross_losses(x, a, b))
            xx = x[[(g, m) in (a, b) for g, m in zip(x.groupset, x.model)]]
            per = np.array([equivalence(xx[xx.seed == k], a, b)["ratio"] for k in sorted(xx.seed.unique())])
            rows.append(dict(freq="4h", setting="pooled fusion", inputs="PMSE", pair=f"{enc}_{fu}/{enc}_metan",
                             ratio=r, se_r0=se0, se_exact=se1, se_seed=np.sqrt(np.var(per, ddof=1) / len(per))))
    t = pd.DataFrame(rows)
    t["se_exact_seed"] = np.sqrt(t.se_exact ** 2 + t.se_seed ** 2)
    for kind in ("r0", "exact", "exact_seed"):
        se = t[f"se_{kind}"]
        t[f"lo_{kind}"], t[f"hi_{kind}"] = t.ratio - 1.645 * se, t.ratio + 1.645 * se
        t[f"p_{kind}"] = [_tost_p(r, s, margin) if np.isfinite(s) else np.nan for r, s in zip(t.ratio, se)]
        t[f"eq_{kind}"] = t[f"p_{kind}"] < 0.05
        ok = t[f"p_{kind}"].notna()
        t.loc[ok, f"p_{kind}_holm"] = holm(t.loc[ok, f"p_{kind}"].to_numpy())
        t[f"eq_{kind}_holm"] = t[f"p_{kind}_holm"] < 0.05
    # sanity: the r0 construction must reproduce Fig. 2 / review_equivalence.csv
    ref = pd.read_csv(OUT_T / "review_equivalence.csv").set_index(["freq", "setting", "inputs", "pair"])
    m = t.set_index(["freq", "setting", "inputs", "pair"]).join(ref[["ci_lo", "ci_hi", "equivalent"]], how="inner")
    print(f"check vs review_equivalence: {len(m)} pairs, max |lo diff| {np.abs(m.lo_r0 - m.ci_lo).max():.2e}, "
          f"flags equal: {(m.eq_r0 == m.equivalent).all()}")
    t.to_csv(OUT_T / "r8_equiv_exact.csv", index=False)
    tw = t[t.setting != "pooled fusion"]
    for nm, s in (("twins (42)", tw), ("fusion (4)", t[t.setting == "pooled fusion"])):
        print(nm, {c: int(s[c].sum()) for c in ("eq_r0", "eq_r0_holm", "eq_exact", "eq_exact_holm",
                                                 "eq_exact_seed", "eq_exact_seed_holm")})
    pd.set_option("display.width", 250)
    print(t[["freq", "setting", "inputs", "pair", "ratio", "lo_r0", "hi_r0", "lo_exact", "hi_exact", "eq_r0",
             "eq_exact", "eq_exact_holm", "eq_exact_seed", "eq_exact_seed_holm"]].round(4).to_string(index=False))
    return t


if __name__ == "__main__":
    args = sys.argv[1:] or ["lgbm_save"]
    seeds = (0, 1, 2)
    for a in args:
        if a.startswith("seeds:"):
            seeds = tuple(int(s) for s in a.split(":")[1].split(","))
    for a in args:
        name, _, opt = a.partition(":")
        if name == "seeds":
            continue
        kw = {}
        if name in ("lgbm_save", "harsx_save"):
            kw = dict(freqs=tuple(opt.split(",")) if opt else FREQS)
            if name == "lgbm_save":
                kw["seeds"] = seeds
        globals()[name](**kw)
