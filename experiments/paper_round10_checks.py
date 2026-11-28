"""Round-10 checks (internal review round 9): local computations on saved forecasts.

  lgbm_ext      the validation tuning of the lagged pooled LightGBM-S (paper_round9_checks.lgbm_tune)
                selected min_child_samples = 400, the top of its grid, at every frequency; the grid is
                extended to 800/1600/3200 (num_leaves 15/31/63) and the selection redone
                -> tables/r10_lgbm_tune_ext_<freq>.csv; if a new point wins, its forecasts are saved
                   (results/r10_local/final_lgbmtuned2_<freq>_r10, model lgbmst2-pool)
  tuned_tests   best calendar network against the (re-)selected tuned LightGBM -> tables/r10_lgbm_tuned_tests.csv
  mcs_tuned     90% MCS of r8_mcs_bench with the tuned LightGBM added -> tables/r10_mcs_bench_tuned.csv
  social_tost   +-1% equivalence of PMSE against PM (every model of review_altdata) -> tables/r10_social_tost.csv
  calib_k       validation level-calibration constants of all 4-hourly forecasts -> tables/r10_calibration_constants.csv
  joint7        family-wise fee test of all seven combinations against the level-calibrated HAR
                -> tables/r10_econ_joint7.csv
  seed_summary  twin verdicts with and without between-seed variance (from existing CSVs), printed

Run:  .venv/bin/python experiments/paper_round10_checks.py lgbm_ext
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_round7_checks import _load  # noqa: E402
from paper_round9_checks import LGBM_GRID, OUT_T, _aligned  # noqa: E402

FREQS = ("1h", "4h", "1d")
LGBM_EXT_GRID = [dict(num_leaves=nl, min_child_samples=mc) for nl in (15, 31, 63) for mc in (800, 1600, 3200)]
LGBM_TUNED2_DIR = "results/r10_local/final_lgbmtuned2_{f}_r10"
TUNED2 = ("PM", "lgbmst2-pool")


# ============================================================================= 1. lgbm_ext
def lgbm_ext(freqs=FREQS):
    """Validation MSE (summed over coins and folds, seed 0) of the extended grid points; the selection is
    redone over the 12 original points (tables/r9_lgbm_tune_<freq>.csv) plus the 9 new ones."""
    import time
    import src.paper.baselines as B
    from paper_round7_checks import _cfg, _frames
    out_rows = []
    for f in freqs:
        old = pd.read_csv(os.path.join(OUT_T, f"r9_lgbm_tune_{f}.csv"))
        groups = _cfg(f"lgbm_{f}")["groupsets"]["PM"]
        fr, win = _frames(f)
        dss = list(fr)
        Xs = [fr[ds][0].X(groups) for ds in dss]
        Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in B.HAR_LAGS[f][1:]]) for X in Xs]
        nf = len(fr[dss[0]][1])
        rows, store = [], {}
        for gi, g in enumerate(LGBM_EXT_GRID):
            base = dict(B.LGBM_PARAMS)
            B.LGBM_PARAMS.update(g)
            t0, vse, res_all = time.time(), 0.0, []
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
            rows.append(dict(freq=f, grid=len(LGBM_GRID) + gi, **g, val_mse_sum=vse))
            print(f, g, f"val {vse:.4f}", f"{time.time() - t0:.0f}s", flush=True)
        allr = pd.concat([old.drop(columns="selected"), pd.DataFrame(rows)], ignore_index=True)
        best = int(allr.loc[allr.val_mse_sum.idxmin(), "grid"])
        allr["selected"] = allr.grid == best
        allr.to_csv(os.path.join(OUT_T, f"r10_lgbm_tune_ext_{f}.csv"), index=False)
        print(f, "selected", allr.loc[allr.selected].to_dict("records"), flush=True)
        if best >= len(LGBM_GRID):                      # a new point wins: save its forecasts
            gi = best - len(LGBM_GRID)
            out = os.path.join(LGBM_TUNED2_DIR.format(f=f), "preds")
            os.makedirs(out, exist_ok=True)
            json.dump({"config": {"name": f"final_lgbmtuned2_{f}", "note": "paper_round10_checks.lgbm_ext",
                                  "params": LGBM_EXT_GRID[gi]}}, open(os.path.join(os.path.dirname(out), "config.json"), "w"))
            for k, res in enumerate(store[gi]):
                for ds, r in zip(dss, res):
                    frame = fr[ds][0].frame
                    idx = frame.index if frame.index.tz is None else frame.index.tz_convert(None)
                    t = idx.as_unit("ns").asi8
                    y = frame["y_range"].to_numpy(float)
                    np.savez(os.path.join(out, f"{ds}__range__PM__lgbmst2-pool__s0__f{k}.npz"),
                             test_time=t[r["test_rows"]], test_y=y[r["test_rows"]].astype("float32"),
                             test_pred=r["test_pred"].astype("float32"), val_time=t[r["val_rows"]],
                             val_y=y[r["val_rows"]].astype("float32"), val_pred=r["val_pred"].astype("float32"),
                             meta=np.array(json.dumps(dict(freq=f, seed=0, fold=k, model="lgbmst2-pool", **LGBM_EXT_GRID[gi]))))
        out_rows.append(allr)
    return pd.concat(out_rows)


def _tuned_key(f):
    """(run name, key) of the selected tuned LightGBM: the extended-grid winner if one was saved."""
    from paper_tables_final import run_dir
    if run_dir(f"final_lgbmtuned2_{f}") is not None:
        return f"final_lgbmtuned2_{f}", TUNED2
    return f"final_lgbmtuned_{f}", ("PM", "lgbmst-pool")


# ============================================================================= 2. tuned_tests
def tuned_tests(freqs=FREQS):
    """As paper_round9_checks.lgbm_tuned_tests with the selection of the extended grid."""
    from paper_review_checks import cross_losses
    from paper_round7_checks import CAL_NET
    from paper_round8_checks import LGBM_LAG, _ratio_se, _tost_p
    from src.paper.evaluate import dm_cross
    rows = []
    for f in freqs:
        tname, tuned = _tuned_key(f)
        df = _load([f"final_calendar_pool_{f}", f"final_lgbmlags_{f}", tname], lambda g, m: (g, m) in {CAL_NET, LGBM_LAG, tuned})
        df = df[~((df.groupset == CAL_NET[0]) & (df.model == CAL_NET[1]) & (df.seed > 2))]
        df = df[~((df.model == LGBM_LAG[1]) & (df.seed > 0))]
        for a, b in ((CAL_NET, tuned), (tuned, LGBM_LAG)):
            xx = df[[(g, m) in (a, b) for g, m in zip(df.groupset, df.model)]]
            per = np.array([np.mean((A - y) ** 2) / np.mean((Bp - y) ** 2) for y, A, Bp in _aligned(xx, a, b).values()])
            c = dm_cross(xx.assign(task="range"), a, b)
            r, _, se = _ratio_se(*cross_losses(xx, a, b))
            rows.append(dict(freq=f, a=f"{a[1]} {a[0]}", b=f"{b[1]} {b[0]}", mean_ratio=per.mean(),
                             better=int((per < 1).sum()), n=len(per), dm=c["dm_stat"], p=c["p"], pooled_ratio=r,
                             lo=r - 1.645 * se, hi=r + 1.645 * se, p_tost=_tost_p(r, se)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_T, "r10_lgbm_tuned_tests.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    return t


# ============================================================================= 3. mcs_tuned
def mcs_tuned(freqs=FREQS):
    """paper_round8_checks.mcs_bench with the selected tuned LightGBM (seed 0) added to the model set."""
    from arch.bootstrap import MCS
    from paper_review_checks import RUNS, ensemble, load_both
    from paper_round8_checks import HARSX, LGBM_LAG
    rows = []
    for f in freqs:
        tname, tuned = _tuned_key(f)
        names = RUNS[f] + [f"final_calendar_pool_{f}", f"final_calendar2_pool_{f}", f"final_lgbmlags_{f}", f"final_harsx_{f}", tname]
        E = ensemble(load_both(names), seeds=[0, 1, 2])
        E = E[(E.split == "test") & ~E.model.isin(["mean"])]
        E["se"] = (E.pred - E.y) ** 2
        L = E.pivot_table(index="time", columns=["groupset", "model"], values="se", aggfunc="mean").dropna(axis=1)
        drop = {"persistence", "ridge", "garch", "gjr"}
        L = L.loc[:, [c for c in L.columns if c[1] not in drop and not (c[1] == "har" and c[0] != "P")]]
        L = L.T.drop_duplicates().T
        mcs = MCS(L.to_numpy(), size=0.10, reps=2000, block_size=max(2, int(len(L) ** (1 / 3))), seed=0)
        mcs.compute()
        cols = list(L.columns)
        names_in = [f"{cols[i][1]} {cols[i][0]}" for i in mcs.included]
        pv = mcs.pvalues.squeeze()
        p = lambda k: float(pv.get(cols.index(k), np.nan)) if k in cols else np.nan
        rows.append(dict(freq=f, n_models=L.shape[1], n_mcs=len(names_in), members="; ".join(names_in),
                         p_lgbm_tuned=p(tuned), p_lgbm_lag=p(LGBM_LAG), p_harsx=p(HARSX), p_cal_net=p(("PMC", "hl_gattn-pool"))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_T, "r10_mcs_bench_tuned.csv"), index=False)


# ============================================================================= 4. social_tost
def social_tost(margin=0.01):
    """PMSE against PM for every pooled model of review_altdata.csv (8 models x 3 frequencies, seeds as
    saved): coin-averaged loss ratio with the exact delta-method 90% interval, TOST p (+-1%), cross-coin
    DM p, and Holm over the 24 TOST and the 24 DM p-values."""
    from paper_review_checks import cross_losses
    from paper_round8_checks import _ratio_se, _tost_p
    from src.paper.evaluate import dm_cross, holm
    alt = pd.read_csv(os.path.join(OUT_T, "review_altdata.csv"))
    alt = alt[~alt.model.str.endswith("PM/P")]
    rows = []
    for f in ("4h", "1h", "1d"):
        models = set(alt[alt.freq == f].model)
        df = _load([f"final_pool_{f}"], lambda g, m: g in ("PM", "PMSE") and m in models)
        for m in sorted(models):
            a, b = ("PMSE", m), ("PM", m)
            x = df[df.model == m]
            r, _, se = _ratio_se(*cross_losses(x, a, b))
            c = dm_cross(x.assign(task="range"), a, b)
            ref = alt[(alt.freq == f) & (alt.model == m)].iloc[0]
            rows.append(dict(freq=f, model=m, mean_ratio=ref.ratio, better=ref.better, pooled_ratio=r,
                             lo=r - 1.645 * se, hi=r + 1.645 * se, p_tost=_tost_p(r, se, margin), dm=c["dm_stat"], p_dm=c["p"],
                             p_dm_ref=ref.p))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t["p_tost_holm"] = holm(t.p_tost.to_numpy())
    t["p_dm_holm"] = holm(t.p_dm.to_numpy())
    t["equivalent"] = t.p_tost < 0.05
    t["equivalent_holm"] = t.p_tost_holm < 0.05
    t["different_holm"] = t.p_dm_holm < 0.05
    t.to_csv(os.path.join(OUT_T, "r10_social_tost.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    print("equivalent", int(t.equivalent.sum()), "| equivalent after Holm", int(t.equivalent_holm.sum()),
          "| DM-different after Holm", int(t.different_holm.sum()), "(PMSE worse:",
          int((t.different_holm & (t.pooled_ratio > 1)).sum()), ") of", len(t))
    return t


# ============================================================================= 5./6. econ (calib_k, joint7)
def _econ_setup():
    """Variance forecasts of paper_round8_checks.econ_cal (all 4-hourly models, the seven bar-level
    combinations and HAR-RV), with every model level-calibrated on validation data (|lvl rows)."""
    import paper_tables_final as T
    from paper_round8_checks import LGBM_LAG, LGBM_LAG_DIR, _recal
    from src.paper.economic import load_preds, variance_forecasts
    freq = "4h"
    econ = T.ECON + [LGBM_LAG]
    T.COMBOS = T.COMBOS + [("PMC", "hl_gattn-pool"), LGBM_LAG]
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
    return pd.concat([V, L], ignore_index=True), ks


def calib_k():
    """Validation level-calibration constants (variance multiplier k per model, coin and fold) of every
    4-hourly forecast; HAR's are those of i7_econ_level_k.csv (median 1.06, range 0.67-1.24 in the text)."""
    _, ks = _econ_setup()
    ks.to_csv(os.path.join(OUT_T, "r10_calibration_constants.csv"), index=False)
    s = ks.groupby(["groupset", "model"]).k.agg(["median", "min", "max", "count"])
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())
    print("all:", ks.k.median().round(3), ks.k.min().round(3), ks.k.max().round(3))
    h = ks[(ks.groupset == "P") & (ks.model == "har")].k
    print("HAR:", round(h.median(), 3), round(h.min(), 3), round(h.max(), 3))
    return ks


def joint7(reps=1000, seed=0):
    """Studentised max-statistic test (as paper_round7_checks.econ_level) over all seven bar-level
    combination fees (gamma = 10, 5 bp, same moving-block draws): (a) uncalibrated combinations against
    plain HAR and against the level-calibrated HAR, (b) level-calibrated combinations against the
    level-calibrated HAR."""
    from src.paper.economic import BARS_PER_YEAR, fko_fee, vol_target
    n = BARS_PER_YEAR["4h"]
    V, _ = _econ_setup()
    res, series = vol_target(V, "4h")
    T_ = len(series[next(iter(series))])
    coins = sorted({k[0] for k in series})
    rng = np.random.default_rng(seed)
    Lb = max(2, int(T_ ** (1 / 3)))
    nb = -(-T_ // Lb)
    draws = [(rng.integers(0, T_ - Lb + 1, nb)[:, None] + np.arange(Lb)).ravel()[:T_] for _ in range(reps)]
    fee = lambda m, b, ix=slice(None): np.mean([fko_fee(series[(c,) + m][ix], series[(c,) + b][ix], 10) * n * 1e4
                                               for c in coins])
    combos = [k for k in dict.fromkeys(zip(V.groupset, V.model)) if k[1].startswith("combo:")
              and not k[1].endswith("|lvl") and k[1] != "combo:har_rv"]
    assert len(combos) == 7, combos
    rows = []
    for tag, b, sfx in (("har", ("P", "har"), ""), ("harlvl", ("P", "har|lvl"), ""), ("lvl_harlvl", ("P", "har|lvl"), "|lvl")):
        cm = [(g, m + sfx) for g, m in combos]
        pts = np.array([fee(m, b) for m in cm])
        B = np.column_stack([[fee(m, b, ix) for ix in draws] for m in cm])
        sd = B.std(0, ddof=1)
        tc = (B - pts) / sd
        tobs = pts / sd
        mx, mxa = tc.max(1), np.abs(tc).max(1)
        for j, (g, m) in enumerate(cm):
            rows.append(dict(test=tag, groupset=g, model=m, fee=pts[j], lo=np.percentile(B[:, j], 5), hi=np.percentile(B[:, j], 95),
                             t=tobs[j], p_fw_one=float(np.mean(mx >= tobs[j])), p_fw_two=float(np.mean(mxa >= abs(tobs[j]))),
                             p_unadj_one=float(np.mean(tc[:, j] >= tobs[j])), p_unadj_two=float(np.mean(np.abs(tc[:, j]) >= abs(tobs[j])))))
        print(tag, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_T, "r10_econ_joint7.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    return t


# ============================================================================= 7. seed_summary
def seed_summary():
    """Twin verdicts with and without the between-seed variance (r8_equiv_exact.csv, i7_seed_twins.csv)."""
    e = pd.read_csv(os.path.join(OUT_T, "r8_equiv_exact.csv"))
    tw = e[e.setting != "pooled fusion"]
    print("equivalence (42 twins):", {c: int(tw[c].sum()) for c in ("eq_r0", "eq_exact", "eq_exact_holm", "eq_exact_seed", "eq_exact_seed_holm")})
    ph = tw[(tw.freq == "1h") & (tw.setting == "pooled")]
    print("pooled hourly:", len(ph), {c: int(ph[c].sum()) for c in ("eq_r0", "eq_exact", "eq_exact_seed", "eq_exact_seed_holm")})
    print(ph[["inputs", "pair", "ratio", "lo_exact_seed", "hi_exact_seed", "eq_exact_seed_holm"]].round(4).to_string(index=False))
    s = pd.read_csv(os.path.join(OUT_T, "i7_seed_twins.csv"))
    print(s.columns.tolist())
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, _, opt = arg.partition(":")
        globals()[name](*([tuple(opt.split(","))] if opt else []))
