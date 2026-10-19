"""
Checks for internal review round 7 (October 2026), from saved forecasts and raw data only
(checks 5-7 refit cheap baselines on the paper's folds; no network is retrained).

    .venv/bin/python -W ignore experiments/paper_round7_checks.py [name ...]
    .venv/bin/python -W ignore experiments/paper_round7_checks.py lgbm_lags:4h,1d,1h   # frequencies

1. seed_equiv: the +-1% equivalence intervals of the 42 twin pairs (Fig. 2, built by
   paper_review_checks.equivalence) with the seed variance added: the current interval is
   r +- 1.645 se, r = 1 + mean(d) / mean(L_b), se^2 = NW long-run variance of d / n / mean(L_b)^2
   (linear ratio scale, NW bandwidth floor(n^(1/3))). New: se^2 + var_k(r_k) / S, r_k the same ratio
   from seed k alone (S = 3 single-coin, 5 pooled seeds). Also the old construction with a 4x bandwidth.
2. seed_twins: the twin table (final_twins.csv) next to the seed-level tests of review_seed_tests.csv
   (t-test of the S per-seed log ratios) and a combined z test on the pooled-loss ratio with the
   seed-augmented variance of check 1; flags starred (Holm DM) rows that the seed t-test does not confirm.
3. boot_coins: moving-block bootstrap over calendar time (the same resampled blocks for every coin;
   2000 reps, seed 0) instead of the Wilcoxon test over coins, for
   (a) the calibrated-QLIKE comparisons of f6_qlike_wilcoxon.csv; statistic = cross-coin mean of the
       per-coin log ratio of mean QLIKE (the quantity the Wilcoxon test ranks); block = floor(T^(1/3)) bars;
   (b) the RMSD_30 comparisons of f7_rmsd_coins.csv; statistic = cross-coin mean of
       RMSD_30(model) - RMSD_30(HAR); the per-date squared deviations (rv30 - 0.40)^2 of all coins and
       both models are resampled jointly (the rolling volatilities are not recomputed); block =
       floor(T^(1/3)) bars and, because rv30 is a 180-bar rolling statistic, also 180 bars (30 days).
   One-sided p in the direction the text claims (model better; p_claim), two-sided otherwise. Because the
   QLIKE bootstrap distributions are strongly skewed (a few extreme bars), three constructions are
   written: basic bootstrap (centred distribution; p_claim, equal-tailed p_two), symmetric |.| (p_two_sym)
   and percentile method (p_pct_claim, p_pct_two); a conclusion is robust only if they agree.
4. econ_level: 4-hourly volatility targeting (40%, leverage <= 2, 5 bp): (a) RMSD_30 around each coin's
   own mean of rv30 (level removed); (b) a level-recalibrated HAR: HAR's variance forecast times one
   constant per coin and fold chosen so that the strategy's realised volatility on that fold's
   validation block is exactly 40% (validation data only); its statistics, and the fees of every model
   against it, with moving-block bootstrap 90% intervals (as review_econ_ci.csv: 1000 reps, seed 0,
   block floor(T^(1/3))); (c) a studentised max-statistic bootstrap over the five combination fees:
   family-wise (single-step) p-values for H0: fee <= 0 and simultaneous two-sided 90% intervals.
5. hars_inter: HAR-S with weekday x time-of-day interaction dummies (4h: 6 x 7, 1h: 24 x 7 cells),
   single-coin OLS on the training block, the paper's folds; MSE vs HAR, HAR-S and the best calendar
   network (pooled H-LSTM with group attention, PMC, seeds 0-2).
6. har_market: HAR and HAR-S plus the current cross-sectional mean log range of all coins in the sample
   at t (one regressor, or its HAR averages), or BTC's log range; single-coin OLS on the training block.
7. lgbm_lags: pooled LightGBM-S (PM inputs, seed 0, same settings) with HAR-style averages over the HAR
   lags (lags[1:] bars; the current bar is already an input) of its eight input features (price and
   market groups), refitted next to an exact refit of the current model.
8. best_epoch: early-stopping epoch of every network fit (npz meta), median and IQR per model and
   setting, quaternion versus real twin.
Outputs: docs/paper/manuscript/tables/i7_<name>.csv (econ_level also i7_econ_level_joint.csv)
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from paper_tables_final import (COMBOS, COMBOS_BENCH, ECON, LABEL, OUT_T, POOL, TWINS, _add_combos,  # noqa: E402
                                _harrv_preds, label, run_dir)
from src.paper.evaluate import _to_time, holm  # noqa: E402

FREQS = ("4h", "1h", "1d")
CAL_NET = ("PMC", "hl_gattn-pool")          # best calendar network at every frequency (review_calendar_summary)


# ============================================================================= helpers
def _load(names, keep, splits=("test",), seeds=None):
    """Long frame (as evaluate.load_run, plus `split`) of the range-task cells of the given runs whose
    (groupset, model) pass `keep`; only those npz files are read."""
    parts = []
    for name in names:
        d = run_dir(name)
        if d is None:
            continue
        for path in glob.glob(os.path.join(d, "preds", "*__range__*.npz")):
            ds, task, gs, model, seed, fold = os.path.basename(path)[:-4].split("__")
            if not keep(gs, model) or (seeds is not None and int(seed[1:]) not in seeds):
                continue
            z = np.load(path)
            for split in splits:
                parts.append(pd.DataFrame({
                    "dataset": ds, "task": task, "groupset": gs, "model": model, "seed": int(seed[1:]),
                    "fold": int(fold[1:]), "split": split, "time": _to_time(z[f"{split}_time"]),
                    "y": z[f"{split}_y"].astype(float), "pred": z[f"{split}_pred"].astype(float)}))
    return pd.concat(parts, ignore_index=True)


def _nw_lrv(d, lag):
    """Bartlett Newey-West long-run variance of d (as in paper_review_checks.equivalence)."""
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    return var


def _mbb_starts(T, L, reps, seed=0):
    """Block start indices of a moving-block bootstrap (reps x n_blocks)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, T - L + 1, size=(reps, -(-T // L)))


def _mbb_idx(starts_row, L, T):
    return (starts_row[:, None] + np.arange(L)).ravel()[:T]


def _boot_p(stat, boot, direction):
    """Bootstrap p-values from the centred bootstrap distribution boot - stat.
    direction -1: H1 theta < 0; +1: H1 theta > 0; 0: two-sided, equal-tailed (2 x the smaller
    one-sided p, robust to the skewed bootstrap distributions of QLIKE); 'sym': symmetric |c| >= |stat|."""
    c = boot - stat
    if direction == "sym":
        return float(np.mean(np.abs(c) >= abs(stat)))
    if direction < 0:
        return float(np.mean(c <= stat))
    if direction > 0:
        return float(np.mean(c >= stat))
    return min(1.0, 2 * min(float(np.mean(c <= stat)), float(np.mean(c >= stat))))


def _pct_p(boot, direction):
    """Percentile-method p-values (is 0 outside the percentile interval of the bootstrap statistic?);
    differs from _boot_p when the bootstrap distribution is skewed."""
    if direction < 0:
        return float(np.mean(boot >= 0))
    if direction > 0:
        return float(np.mean(boot <= 0))
    return min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))


def _cfg(name):
    import yaml
    return yaml.safe_load(open(f"configs/paper/final/{name}.yaml"))


def _frames(freq):
    """Datasets and folds of a frequency exactly as the runner builds them (single_{freq} config)."""
    from src.paper.data import make_dataset, walk_forward_folds
    cfg = _cfg(f"single_{freq}")
    fc = cfg["folds"]
    out = {}
    for ds in cfg["datasets"]:
        coin = ds.split("_")[0]
        d = make_dataset(coin, freq, start=cfg.get("start"))
        out[ds] = (d, walk_forward_folds(d.frame.index, fc["n"], fc["test_days"], fc["val_days"], end=fc.get("end")))
    return out, cfg["window"][freq]


def _saved_test(freq, keys, names=None, seeds=None):
    """Seed-averaged saved test forecasts of (groupset, model) keys: dataset -> DataFrame(time x key)."""
    names = names or [f"final_{x}_{freq}" for x in ("single", "pool", "har_season", "lgbm", "calendar_pool")]
    df = _load(names, lambda g, m: (g, m) in keys, seeds=seeds)
    e = df.groupby(["dataset", "groupset", "model", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
    out = {}
    for ds, g in e.groupby("dataset"):
        P = g.pivot_table(index="time", columns=["groupset", "model"], values="pred")
        P.columns = [tuple(c) for c in P.columns]
        P["y"] = g.groupby("time").y.first()
        out[ds] = P
    return out


def _ratio_summary(rows, freq, a, b):
    """Mean over coins of MSE(a) / MSE(b) and coins with a < b, from per-coin rows {col: mse}."""
    r = np.array([x[a] / x[b] for x in rows])
    return dict(freq=freq, a=a, b=b, mean=r.mean(), min=r.min(), max=r.max(), better=int((r < 1).sum()), n=len(r))


# ============================================================================= 1. seed_equiv
def seed_equiv(margin=0.01):
    from paper_review_checks import cross_losses, equivalence
    rows = []
    for f in FREQS:
        keep = lambda g, m: g in ("P", "PMSE") and m.replace(POOL, "") in {x for t in TWINS for x in t}
        df = _load([f"final_single_{f}", f"final_pool_{f}"], keep)
        for tag, sfx in (("single", ""), ("pooled", POOL)):
            for gs in ("P", "PMSE"):
                for q, r in TWINS:
                    a, b = (gs, q + sfx), (gs, r + sfx)
                    x = df[[(g, m) in (a, b) for g, m in zip(df.groupset, df.model)]]
                    if x.model.nunique() < 2:
                        continue
                    LA, LB = cross_losses(x, a, b)
                    d = (LA - LB).to_numpy()
                    n, lb = len(d), LB.mean()
                    lag = int(np.floor(n ** (1 / 3)))
                    ratio = 1 + d.mean() / lb
                    se_nw = np.sqrt(_nw_lrv(d, lag) / n) / lb
                    se_nw4 = np.sqrt(_nw_lrv(d, 4 * lag) / n) / lb
                    seeds = sorted(x.seed.unique())
                    per = np.array([equivalence(x[x.seed == k], a, b)["ratio"] for k in seeds])
                    se_seed = np.sqrt(np.var(per, ddof=1) / len(per))
                    se_new = np.sqrt(se_nw ** 2 + se_seed ** 2)
                    rec = dict(freq=f, setting=tag, inputs=gs, pair=f"{q}/{r}", ratio=ratio, n_seeds=len(per),
                               seed_min=per.min(), seed_max=per.max(), se_nw=se_nw, se_nw4=se_nw4, se_seed=se_seed)
                    for kind, se in (("old", se_nw), ("nw4", se_nw4), ("new", se_new)):
                        lo, hi = ratio - 1.645 * se, ratio + 1.645 * se
                        rec.update({f"lo_{kind}": lo, f"hi_{kind}": hi,
                                    f"eq_{kind}": bool(lo > 1 - margin and hi < 1 + margin)})
                    rows.append(rec)
        print(f, flush=True)
    t = pd.DataFrame(rows)
    # sanity: the old construction must reproduce Fig. 2 (fig_forest.csv)
    ff = pd.read_csv(OUT_T / "fig_forest.csv").set_index(["freq", "setting", "inputs", "pair"])
    m = t.set_index(["freq", "setting", "inputs", "pair"]).join(ff[["ci_lo", "ci_hi", "equivalent"]], how="inner")
    print(f"check vs fig_forest: {len(m)} pairs, max |lo diff| {np.abs(m.lo_old - m.ci_lo).max():.2e}, "
          f"max |hi diff| {np.abs(m.hi_old - m.ci_hi).max():.2e}, equivalence flags equal: {(m.eq_old == m.equivalent).all()}")
    t.to_csv(OUT_T / "i7_seed_equiv.csv", index=False)
    s = t.groupby(["freq", "setting"], sort=False).agg(pairs=("pair", "size"), eq_old=("eq_old", "sum"),
                                                      eq_nw4=("eq_nw4", "sum"), eq_new=("eq_new", "sum"))
    s["median_se_new_over_old"] = (np.sqrt(t.se_nw ** 2 + t.se_seed ** 2) / t.se_nw).groupby([t.freq, t.setting]).median()
    pd.set_option("display.width", 250)
    print(t[["freq", "setting", "inputs", "pair", "ratio", "lo_old", "hi_old", "lo_new", "hi_new", "eq_old", "eq_nw4",
             "eq_new"]].round(4).to_string(index=False))
    print(s.round(2).to_string())
    print("total equivalent: old", int(t.eq_old.sum()), "| NW 4x", int(t.eq_nw4.sum()), "| seed-augmented",
          int(t.eq_new.sum()), "of", len(t))
    return t


# ============================================================================= 2. seed_twins
def seed_twins():
    tw = pd.read_csv(OUT_T / "final_twins.csv")
    st = pd.read_csv(OUT_T / "review_seed_tests.csv")
    st = st[st.inputs == "PMSE"]
    if not (OUT_T / "i7_seed_equiv.csv").exists():
        seed_equiv()
    eq = pd.read_csv(OUT_T / "i7_seed_equiv.csv")
    eq = eq[eq.inputs == "PMSE"]
    inv = {f"{LABEL[q]} / {LABEL[r]}": f"{q}/{r}" for q, r in TWINS}
    rows = []
    for r in tw.itertuples():
        f, s = r.setting.split(" ")[1], r.setting.split(" ")[0]
        k = inv[r.pair]
        x = st[(st.freq == f) & (st.setting == s) & (st.pair == k)].iloc[0]
        e = eq[(eq.freq == f) & (eq.setting == s) & (eq.pair == k)].iloc[0]
        se = np.sqrt(e.se_nw ** 2 + e.se_seed ** 2)
        z = (e.ratio - 1) / se
        rows.append(dict(setting=r.setting, pair=r.pair, ratio=r.ratio, dm=r.dm, p_holm=r.p_holm,
                         star=bool(r.p_holm < 0.05), seed_min=r.seed_min, seed_max=r.seed_max, n_seeds=int(x.n_seeds),
                         seed_ratios=x.seed_ratios, seed_same_sign=int(x.seed_same_sign),
                         p_seed_ttest=x.ttest_logratio_p, p_sign=x.sign_test_p,
                         pooled_ratio=e.ratio, z_combined=z, p_combined=2 * (1 - stats.norm.cdf(abs(z)))))
    t = pd.DataFrame(rows)
    t["p_seed_ttest_holm"] = holm(t.p_seed_ttest)
    t["p_combined_holm"] = holm(t.p_combined)
    t["flag"] = t.star & (t.p_seed_ttest >= 0.05)       # starred, not confirmed by the seed t-test
    t.to_csv(OUT_T / "i7_seed_twins.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.drop(columns=["seed_ratios", "pooled_ratio"]).round(4).to_string(index=False))
    n = int(t.star.sum())
    print(f"starred (Holm DM < .05): {n} | of these seed t-test p < .05: {int((t.star & (t.p_seed_ttest < .05)).sum())}"
          f" | Holm(seed t) < .05: {int((t.star & (t.p_seed_ttest_holm < .05)).sum())}"
          f" | combined z p < .05: {int((t.star & (t.p_combined < .05)).sum())}"
          f" | Holm(combined) < .05: {int((t.star & (t.p_combined_holm < .05)).sum())}")
    print("flagged:", "; ".join(f"{a} {b}" for a, b in zip(t[t.flag].setting, t[t.flag].pair)))
    return t


# ============================================================================= 3. boot_coins
def _econ_V(freq="4h"):
    """Calibrated variance forecasts of the economic table (test rows, with the HAR combinations and
    HAR-RV) and of the validation rows, exactly as fresh7.rmsd_coins / economic_table build them."""
    from src.paper.economic import load_preds, variance_forecasts
    dirs = [run_dir(f"final_{x}_{freq}") for x in ("single", "pool", "harx", "har_season", "lgbm", "calendar_pool")]
    keep = lambda d: d[[(g, m) in ECON for g, m in zip(d.groupset, d.model)]]
    test = pd.concat([keep(load_preds(d)) for d in dirs if d])
    val = pd.concat([keep(load_preds(d, split="val")) for d in dirs if d])
    rt, rv = _harrv_preds(test, val, freq)
    test, val = pd.concat([test, rt], ignore_index=True), pd.concat([val, rv], ignore_index=True)
    Vv = variance_forecasts(val, val)
    return _add_combos(variance_forecasts(test, val), Vv), Vv


def _rv30(V, series, n, days=30):
    """(groupset, model) -> DataFrame (time x dataset) of the rolling `days`-day realised volatility of
    each targeted position, per coin as in fresh7.rmsd_coins."""
    win = int(n / 365 * days)
    times = {k: g.sort_values("time").time.to_numpy() for k, g in V.groupby(["dataset", "groupset", "model"])}
    per = {}
    for (ds, gs, m), rp in series.items():
        per.setdefault((gs, m), {})[ds] = pd.Series(rp, index=times[(ds, gs, m)]).rolling(win).std() * np.sqrt(n)
    return {k: pd.DataFrame(d).sort_index() for k, d in per.items()}, win


QL_CLAIM = {  # direction the text claims for each f6 QLIKE pair (-1: a better, 0: no difference claimed)
    ("4h", "LightGBM-S / H-LSTM (gattn) PMC"): 0, ("1h", "LightGBM-S / H-LSTM (gattn) PMC"): 0,
    ("1d", "LightGBM-S / H-LSTM (gattn) PMC"): -1, ("1d", "LightGBM-S / H-Transformer (concat) PMC"): 0,
    ("4h", "Transformer pooled PMSE / HAR"): -1, ("1h", "Transformer pooled PMSE / HAR"): -1,
    ("4h", "H-LSTM (gattn) PMC / HAR"): -1, ("1h", "H-LSTM (gattn) PMC / HAR"): -1,
    ("1d", "HAR-S / HAR"): -1, ("4h", "Transformer pooled / H-Transformer (meta) pooled"): -1,
    ("1h", "Transformer pooled / H-Transformer (meta) pooled"): -1,
}


def boot_coins(reps=2000, seed=0):
    from paper_fresh6_checks import QL_PAIRS
    from paper_review_checks import calibrated_variance, ensemble, qlike_loss
    rows = []
    f6 = pd.read_csv(OUT_T / "f6_qlike_wilcoxon.csv").set_index(["freq", "pair"])
    keys = {k for ab in QL_PAIRS.values() for k in ab}
    for f in FREQS:
        names = [f"final_{x}_{f}" for x in ("single", "pool", "har_season", "lgbm", "calendar_pool", "calendar2_pool")]
        E = ensemble(_load(names, lambda g, m: (g, m) in keys, splits=("test", "val"), seeds=[0, 1, 2]))
        V = calibrated_variance(E)
        V["ql"] = qlike_loss(V.s2, V.h)
        Q = {k: g.pivot_table(index="time", columns="dataset", values="ql") for k, g in V.groupby(["groupset", "model"])}
        T = len(next(iter(Q.values())))
        L = max(2, int(T ** (1 / 3)))
        starts = _mbb_starts(T, L, reps, seed)
        for name, (a, b) in QL_PAIRS.items():
            if a not in Q or b not in Q:
                continue
            A, B = Q[a], Q[b].reindex(index=Q[a].index, columns=Q[a].columns)
            assert len(A) == T and not np.isnan(A.values).any() and not np.isnan(B.values).any()
            A, B = A.to_numpy(), B.to_numpy()
            ratio = A.mean(0) / B.mean(0)
            stat = np.log(ratio).mean()
            boot = np.array([np.log(A[ix].mean(0) / B[ix].mean(0)).mean()
                             for ix in (_mbb_idx(s, L, T) for s in starts)])
            claim = QL_CLAIM.get((f, name), 0)
            old = f6.loc[(f, name)]
            rows.append(dict(part="qlike", freq=f, comparison=name, statistic="mean_c log(QL_a/QL_b)", stat=stat,
                             ratio_mean=ratio.mean(), ratio_f6=old.ratio, better=int((ratio < 1).sum()), n=len(ratio),
                             block=L, claim=claim, p_one=_boot_p(stat, boot, -1 if stat < 0 else 1),
                             p_two=_boot_p(stat, boot, 0), p_two_sym=_boot_p(stat, boot, "sym"),
                             p_claim=_boot_p(stat, boot, claim), p_pct_claim=_pct_p(boot, claim),
                             p_pct_two=_pct_p(boot, 0),
                             lo90=np.percentile(2 * stat - boot, 5), hi90=np.percentile(2 * stat - boot, 95),
                             p_wilcoxon=old.p_wilcoxon))
        print(f, flush=True)
    # (b) RMSD_30 of the 4-hourly volatility-targeting exercise against HAR
    from src.paper.economic import BARS_PER_YEAR, vol_target
    V, _ = _econ_V("4h")
    _, series = vol_target(V, "4h")
    n = BARS_PER_YEAR["4h"]
    R, win = _rv30(V, series, n)
    f7 = pd.read_csv(OUT_T / "f7_rmsd_coins.csv").set_index("model")
    H = R[("P", "har")]
    coins = list(H.columns)
    Hd = ((H[coins] - 0.40) ** 2).to_numpy()
    T = len(Hd)
    for L in (max(2, int(T ** (1 / 3))), win):
        starts = _mbb_starts(T, L, reps, seed)
        idx = [_mbb_idx(s, L, T) for s in starts]
        for gs, m in ECON + [("P", "har_rv"), ("P", "har_rv_s")] + [(g, "combo:" + x) for g, x in COMBOS + COMBOS_BENCH]:
            if (gs, m) == ("P", "har") or (gs, m) not in R:
                continue
            X = R[(gs, m)].reindex(index=H.index)[coins]
            Xd = ((X - 0.40) ** 2).to_numpy()
            stat = np.mean(np.sqrt(np.nanmean(Xd, 0)) - np.sqrt(np.nanmean(Hd, 0)))
            boot = np.array([np.mean(np.sqrt(np.nanmean(Xd[ix], 0)) - np.sqrt(np.nanmean(Hd[ix], 0))) for ix in idx])
            name = ("HAR + " + label(gs, m[6:])) if m.startswith("combo:") else label(gs, m)
            old = f7.loc[name]
            rows.append(dict(part="rmsd30", freq="4h", comparison=f"{name} - HAR", statistic="mean_c RMSD30 diff",
                             stat=stat, ratio_mean=np.nan, ratio_f6=old.rmsd_mean - old.har_mean,
                             better=int(old.lower), n=int(old.n), block=L, claim=-1,
                             p_one=_boot_p(stat, boot, -1), p_two=_boot_p(stat, boot, 0),
                             p_two_sym=_boot_p(stat, boot, "sym"), p_claim=_boot_p(stat, boot, -1),
                             p_pct_claim=_pct_p(boot, -1), p_pct_two=_pct_p(boot, 0),
                             lo90=np.percentile(2 * stat - boot, 5), hi90=np.percentile(2 * stat - boot, 95),
                             p_wilcoxon=old.p_wilcoxon))
        print("rmsd block", L, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "i7_boot_coins.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["freq", "comparison", "stat", "ratio_mean", "ratio_f6", "better", "block", "claim", "p_claim", "p_two",
            "p_two_sym", "p_pct_claim", "p_pct_two", "p_wilcoxon"]
    print(t[t.part == "qlike"][cols].round(4).to_string(index=False))
    print(t[t.part == "rmsd30"][cols[:3] + cols[4:]].round(4).to_string(index=False))
    return t


# ============================================================================= 4. econ_level
def _recal_har(V, Vv, freq, target=0.40, w_max=2.0, cost=0.0005):
    """HAR variance x k (one k per coin and fold) with k set so that the targeted strategy's realised
    volatility on that fold's validation block equals the target. Validation data only."""
    from src.paper.economic import BARS_PER_YEAR
    n = BARS_PER_YEAR[freq]
    tgt = target / np.sqrt(n)
    har, har_v = (x[(x.groupset == "P") & (x.model == "har")] for x in (V, Vv))
    out, ks = [], []
    for (ds, fold), h in har.groupby(["dataset", "fold"]):
        v = har_v[(har_v.dataset == ds) & (har_v.fold == fold)].sort_values("time")
        simple, var = np.expm1(v.r.to_numpy()), v["var"].to_numpy()

        def gap(logk):
            w = np.minimum(w_max, tgt / np.sqrt(np.exp(logk) * var))
            rp = w * simple - cost * np.abs(np.diff(w, prepend=w[0]))
            return np.std(rp) * np.sqrt(n) - target
        k = np.exp(optimize.brentq(gap, -8, 8))
        ks.append(dict(dataset=ds, fold=fold, k=k))
        out.append(h.assign(model="har_lvl", var=k * h["var"].to_numpy()))
    return pd.concat([V] + out, ignore_index=True), pd.DataFrame(ks)


def econ_level_k():
    """Only the level-calibration constants of HAR (one per coin and fold) -> i7_econ_level_k.csv."""
    V, Vv = _econ_V("4h")
    _, ks = _recal_har(V, Vv, "4h")
    ks.to_csv(OUT_T / "i7_econ_level_k.csv", index=False)
    print("median", round(ks.k.median(), 3), "range", round(ks.k.min(), 3), round(ks.k.max(), 3))
    return ks


def econ_level(reps=1000, seed=0):
    from src.paper.economic import BARS_PER_YEAR, fko_fee, vol_target
    freq = "4h"
    n = BARS_PER_YEAR[freq]
    V, Vv = _econ_V(freq)
    V, ks = _recal_har(V, Vv, freq)
    ks.to_csv(OUT_T / "i7_econ_level_k.csv", index=False)
    print("level factor k (HAR var multiplier): median", round(ks.k.median(), 3), "IQR",
          ks.k.quantile([.25, .75]).round(3).tolist(), "range", round(ks.k.min(), 3), round(ks.k.max(), 3), flush=True)
    res, series = vol_target(V, freq)
    R, _ = _rv30(V, series, n)
    T = len(series[next(iter(series))])
    assert all(len(v) == T for v in series.values()), "unequal test lengths"
    coins = sorted({k[0] for k in series})
    models = [(gs, m) for gs, m in ECON + [("P", "har_rv"), ("P", "har_rv_s"), ("P", "har_lvl")] +
              [(g, "combo:" + x) for g, x in COMBOS + COMBOS_BENCH] if (coins[0], gs, m) in series]
    # moving-block draws generated exactly as paper_review_checks.main_econci (band 0, short blocks)
    rng = np.random.default_rng(seed)
    L = max(2, int(T ** (1 / 3)))
    nb = -(-T // L)
    draws = [(rng.integers(0, T - L + 1, nb)[:, None] + np.arange(L)).ravel()[:T] for _ in range(reps)]
    fee = lambda m, b, ix=slice(None): np.mean([fko_fee(series[(c,) + m][ix], series[(c,) + b][ix], 10) * n * 1e4
                                               for c in coins])
    boots = {}
    rows = []
    for gs, m in models:
        g = res[(res.groupset == gs) & (res.model == m)]
        Rm = R[(gs, m)]
        rec = dict(groupset=gs, model=m, real_vol=g.ann_vol.mean(), vol_err=g.vol_err.mean(),
                   rmsd30=float(np.sqrt(np.nanmean((Rm.to_numpy() - 0.40) ** 2))),
                   rmsd30_demeaned=float(np.sqrt(np.nanmean((Rm - Rm.mean()).to_numpy() ** 2))),
                   rmsd30_coin=float(np.mean(np.sqrt(np.nanmean((Rm.to_numpy() - 0.40) ** 2, 0)))),
                   rmsd30_demeaned_coin=float(np.mean(np.sqrt(np.nanmean((Rm - Rm.mean()).to_numpy() ** 2, 0)))),
                   rv30_level=float(np.nanmean(Rm.to_numpy())), sharpe=g.sharpe.mean())
        for tag, b in (("har", ("P", "har")), ("harlvl", ("P", "har_lvl"))):
            if (gs, m) == b:
                continue
            bs = np.array([fee((gs, m), b, ix) for ix in draws])
            boots[(tag, gs, m)] = bs
            pt = fee((gs, m), b)
            rec.update({f"fee_{tag}": pt, f"fee_{tag}_lo": np.percentile(bs, 5), f"fee_{tag}_hi": np.percentile(bs, 95),
                        f"fee_{tag}_pos": int(sum(fko_fee(series[(c, gs, m)], series[(c,) + b], 10) > 0 for c in coins))})
        rows.append(rec)
        print(gs, m, flush=True)
    t = pd.DataFrame(rows)
    # sanity: the combination fees vs HAR and their intervals must reproduce review_econ_ci.csv
    ci = pd.read_csv(OUT_T / "review_econ_ci.csv")
    ci = ci[(ci.band == 0) & (ci.block == L) & (ci.stat == "fee10")]
    chk = t.merge(ci[["groupset", "model", "point", "lo", "hi"]], on=["groupset", "model"])
    print("check vs review_econ_ci (fee10, band 0):", len(chk), "models, max |point diff|",
          f"{np.abs(chk.fee_har - chk.point).max():.3g}, max |lo diff| {np.abs(chk.fee_har_lo - chk.lo).max():.3g}, "
          f"max |hi diff| {np.abs(chk.fee_har_hi - chk.hi).max():.3g}")
    # sanity: plain HAR's pooled RMSD_30 against Table 13 (final_econ_4h.csv)
    e = pd.read_csv(OUT_T / "final_econ_4h.csv").set_index(["groupset", "model"])
    t["rmsd30_table"] = [e.rmsd30.get((g, m), np.nan) for g, m in zip(t.groupset, t.model)]
    t["fee_table"] = [e.fee.get((g, m), np.nan) for g, m in zip(t.groupset, t.model)]
    t.to_csv(OUT_T / "i7_econ_level.csv", index=False)
    # (c) studentised max-statistic over the five combination fees (single-step, Romano-Wolf style)
    jrows = []
    for tag in ("har", "harlvl"):
        cm = [(g, "combo:" + x) for g, x in COMBOS + COMBOS_BENCH[:2]]
        pts = np.array([t[(t.groupset == g) & (t.model == m)][f"fee_{tag}"].iloc[0] for g, m in cm])
        B = np.column_stack([boots[(tag, g, m)] for g, m in cm])
        sd = B.std(0, ddof=1)
        tc = (B - pts) / sd                                  # centred, studentised bootstrap statistics
        tobs = pts / sd
        mx, mxa = tc.max(1), np.abs(tc).max(1)
        c90 = np.quantile(mxa, 0.90)
        c95_one = np.quantile(mx, 0.95)
        for j, (g, m) in enumerate(cm):
            jrows.append(dict(bench=tag, groupset=g, model=m, fee=pts[j], sd=sd[j], t=tobs[j],
                              p_fw_one=float(np.mean(mx >= tobs[j])),          # H0: fee <= 0, family-wise
                              p_fw_two=float(np.mean(mxa >= abs(tobs[j]))),
                              p_unadj_one=float(np.mean(tc[:, j] >= tobs[j])),
                              sim90_lo=pts[j] - c90 * sd[j], sim90_hi=pts[j] + c90 * sd[j],
                              sim95_lower=pts[j] - c95_one * sd[j], crit_two90=c90, crit_one95=c95_one,
                              p_global_one=float(np.mean(mx >= tobs.max()))))
    j = pd.DataFrame(jrows)
    j.to_csv(OUT_T / "i7_econ_level_joint.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["groupset", "model", "real_vol", "vol_err", "rmsd30", "rmsd30_table", "rmsd30_demeaned", "rmsd30_demeaned_coin",
            "fee_har", "fee_har_lo", "fee_har_hi", "fee_harlvl", "fee_harlvl_lo", "fee_harlvl_hi", "fee_harlvl_pos"]
    print(t[show].round(4).to_string(index=False))
    print(j.round(3).to_string(index=False))
    return t, j


# ============================================================================= 5. hars_inter
def _interactions(index, step):
    """Dummies (one cell dropped) for each (time of day, day of week) cell of bar t+1."""
    nxt = index + step
    key = (nxt.hour * 60 + nxt.minute) * 10 + nxt.dayofweek
    return np.column_stack([(key == v).astype(float) for v in np.unique(key)[1:]])


def hars_inter():
    from src.paper.baselines import HAR_LAGS, _calendar, _har_design, _ols
    rows = []
    for f in ("4h", "1h"):
        fr, win = _frames(f)
        saved = _saved_test(f, {("P", "har"), ("PMSE", "hars"), CAL_NET}, seeds=[0, 1, 2])
        for ds, (d, folds) in fr.items():
            y = d.frame["y_range"].to_numpy(float)
            H = _har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[f])
            C, I = _calendar(d.frame.index, d.step), _interactions(d.frame.index, d.step)
            pred = {}
            for name, D in (("har", H), ("hars", np.column_stack([H, C])), ("hars_int", np.column_stack([H, I]))):
                out = pd.Series(np.nan, index=d.frame.index)
                for fo in folds:
                    tr, te = fo["train"], fo["test"]
                    te = te[te >= win - 1]
                    ok = tr[~np.isnan(D[tr]).any(1)]
                    b = _ols(D[ok], y[ok])
                    out.iloc[te] = b[0] + D[te] @ b[1:]
                pred[name] = out
            S = saved[ds]
            P = pd.DataFrame({k: v.reindex(S.index).to_numpy() for k, v in pred.items()}, index=S.index)
            yy = S.y.to_numpy()
            mse = {k: np.mean((P[k].to_numpy() - yy) ** 2) for k in P}
            mse.update(har_saved=np.mean((S[("P", "har")] - yy) ** 2), hars_saved=np.mean((S[("PMSE", "hars")] - yy) ** 2),
                       net=np.mean((S[CAL_NET] - yy) ** 2), n_dummies=I.shape[1])
            rows.append(dict(freq=f, dataset=ds, **mse))
        print(f, flush=True)
    raw = pd.DataFrame(rows)
    out = []
    for f, g in raw.groupby("freq", sort=False):
        recs = g.to_dict("records")
        for a, b in (("har", "har_saved"), ("hars", "hars_saved"), ("hars", "har"), ("hars_int", "har"),
                     ("hars_int", "hars"), ("net", "har"), ("net", "hars"), ("net", "hars_int")):
            out.append(dict(**_ratio_summary(recs, f, a, b), n_dummies=int(g.n_dummies.iloc[0])))
    t = pd.DataFrame(out)
    t["a"] = t.a.replace({"net": "H-LSTM (group attn., pooled), PMC"})
    t.to_csv(OUT_T / "i7_hars_inter.csv", index=False)
    print(t.round(4).to_string(index=False))
    return t


# ============================================================================= 6. har_market
def har_market():
    from src.paper.baselines import HAR_LAGS, _calendar, _har_design, _ols
    rows = []
    for f in FREQS:
        fr, win = _frames(f)
        lr = pd.concat({ds: d.frame["log_range"] for ds, (d, _) in fr.items()}, axis=1).sort_index()
        mkt = lr.mean(axis=1)                    # coins in their samples at t only (NaN elsewhere)
        btc = lr[f"btc_{f}"]
        saved = _saved_test(f, {("P", "har"), ("PMSE", "hars")}, names=[f"final_single_{f}", f"final_har_season_{f}"])
        for ds, (d, folds) in fr.items():
            idx = d.frame.index
            y = d.frame["y_range"].to_numpy(float)
            H = _har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[f])
            C = _calendar(idx, d.step)
            m1 = mkt.reindex(idx).to_numpy()[:, None]
            mH = _har_design(mkt.to_numpy(), HAR_LAGS[f])
            mH = pd.DataFrame(mH, index=mkt.index).reindex(idx).to_numpy()
            b1 = btc.reindex(idx).to_numpy()[:, None]
            designs = {"har": H, "har+mkt": np.column_stack([H, m1]), "har+mktHAR": np.column_stack([H, mH]),
                       "har+btc": np.column_stack([H, b1]), "hars": np.column_stack([H, C]),
                       "hars+mkt": np.column_stack([H, C, m1]), "hars+btc": np.column_stack([H, C, b1])}
            S = saved[ds]
            yy = S.y.to_numpy()
            mse = dict(har_saved=np.mean((S[("P", "har")] - yy) ** 2), hars_saved=np.mean((S[("PMSE", "hars")] - yy) ** 2))
            for name, D in designs.items():
                out = pd.Series(np.nan, index=idx)
                for fo in folds:
                    tr, te = fo["train"], fo["test"]
                    te = te[te >= win - 1]
                    ok = tr[~np.isnan(D[tr]).any(1)]
                    b = _ols(D[ok], y[ok])
                    out.iloc[te] = b[0] + D[te] @ b[1:]
                p = out.reindex(S.index).to_numpy()
                assert not np.isnan(p).any(), (ds, name)
                mse[name] = np.mean((p - yy) ** 2)
            rows.append(dict(freq=f, dataset=ds, **mse))
        print(f, flush=True)
    raw = pd.DataFrame(rows)
    out = []
    for f, g in raw.groupby("freq", sort=False):
        recs = g.to_dict("records")
        for a, b in (("har", "har_saved"), ("hars", "hars_saved"), ("har+mkt", "har"), ("har+mktHAR", "har"),
                     ("har+btc", "har"), ("hars", "har"), ("hars+mkt", "har"), ("hars+btc", "har"),
                     ("hars+mkt", "hars"), ("hars+btc", "hars")):
            out.append(_ratio_summary(recs, f, a, b))
    t = pd.DataFrame(out)
    t.to_csv(OUT_T / "i7_har_market.csv", index=False)
    print(t.round(4).to_string(index=False))
    return t


# ============================================================================= 7. lgbm_lags
def lgbm_lags(freqs=FREQS):
    import time
    import lightgbm  # noqa: F401  (load its OpenMP runtime first, as the runner does)
    from src.paper.baselines import HAR_LAGS, predict_baseline_pooled
    rows = []
    for f in freqs:
        groups = _cfg(f"lgbm_{f}")["groupsets"]["PM"]
        fr, win = _frames(f)
        dss = list(fr)
        Xs = [fr[ds][0].X(groups) for ds in dss]
        # HAR-style averages of every input feature over the longer HAR lags (current value = lag 1, already in X)
        Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS[f][1:]]) for X in Xs]
        preds = {"lgbms_refit": {}, "lgbms_lags": {}}
        for k in range(len(fr[dss[0]][1])):
            for name, XX in (("lgbms_refit", Xs), ("lgbms_lags", Xl)):
                t0 = time.time()
                res = predict_baseline_pooled("lgbms", [fr[ds][0].frame for ds in dss], "range",
                                              [fr[ds][1][k] for ds in dss], win, f, XX, seed=0)
                for ds, r in zip(dss, res):
                    idx = fr[ds][0].frame.index
                    preds[name].setdefault(ds, []).append(pd.Series(r["test_pred"], index=idx[r["test_rows"]]))
                print(f, "fold", k, name, f"{time.time() - t0:.0f}s", flush=True)
        saved = _saved_test(f, {("P", "har"), ("PM", "lgbms-pool"), CAL_NET},
                            names=[f"final_single_{f}", f"final_lgbm_{f}", f"final_calendar_pool_{f}"], seeds=[0, 1, 2])
        recs = []
        for ds in dss:
            S = saved[ds]
            yy = S.y.to_numpy()
            p = {k: pd.concat(v[ds]).reindex(S.index).to_numpy() for k, v in preds.items()}
            recs.append(dict(dataset=ds, har=np.mean((S[("P", "har")] - yy) ** 2),
                             lgbms_saved=np.mean((S[("PM", "lgbms-pool")] - yy) ** 2),
                             net=np.mean((S[CAL_NET] - yy) ** 2),
                             maxdiff_refit=float(np.max(np.abs(p["lgbms_refit"] - S[("PM", "lgbms-pool")].to_numpy()))),
                             **{k: np.mean((v - yy) ** 2) for k, v in p.items()}))
        print(f, "refit vs saved LightGBM-S: max |pred diff|", round(max(r["maxdiff_refit"] for r in recs), 6), flush=True)
        for a, b in (("lgbms_refit", "lgbms_saved"), ("lgbms_saved", "har"), ("lgbms_lags", "har"),
                     ("lgbms_lags", "lgbms_saved"), ("net", "lgbms_saved"), ("net", "lgbms_lags")):
            rows.append(_ratio_summary(recs, f, a, b))
    t = pd.DataFrame(rows)
    t["a"] = t.a.replace({"net": "H-LSTM (group attn., pooled), PMC"})
    t["note"] = "1h not run" if "1h" not in freqs else ""
    t.to_csv(OUT_T / "i7_lgbm_lags.csv", index=False)
    print(t.round(4).to_string(index=False))
    return t


# ============================================================================= 8. best_epoch
def best_epoch():
    rows = []
    for f in FREQS:
        for tag, name in (("single", f"final_single_{f}"), ("pooled", f"final_pool_{f}")):
            d = run_dir(name)
            for path in glob.glob(os.path.join(d, "preds", "*__range__*.npz")):
                ds, _, gs, model, seed, fold = os.path.basename(path)[:-4].split("__")
                if tag == "pooled" and not ds.startswith("btc_"):      # one pooled fit writes all 15 coins
                    continue
                meta = json.loads(str(np.load(path)["meta"]))
                if "best_epoch" not in meta:                           # baselines
                    continue
                rows.append(dict(freq=f, setting=tag, inputs=gs, model=model.replace(POOL, ""), dataset=ds,
                                 seed=int(seed[1:]), fold=int(fold[1:]), best_epoch=meta["best_epoch"],
                                 epochs=meta["epochs"]))
        print(f, flush=True)
    raw = pd.DataFrame(rows)
    q = lambda p: (lambda v: float(np.quantile(v, p)))
    s = raw.groupby(["freq", "setting", "inputs", "model"]).agg(
        fits=("best_epoch", "size"), median=("best_epoch", "median"), q25=("best_epoch", q(.25)),
        q75=("best_epoch", q(.75)), mean_epochs=("epochs", "mean"), max_epochs=("epochs", "max"),
        share_le2=("best_epoch", lambda v: float(np.mean(v <= 2)))).reset_index()
    s["kind"] = "model"
    # quaternion vs real twin on matched fits (same coin, seed, fold)
    tw = []
    for (f, st, gs), g in raw.groupby(["freq", "setting", "inputs"]):
        for qm, rm in TWINS:
            a = g[g.model == qm].set_index(["dataset", "seed", "fold"]).best_epoch
            b = g[g.model == rm].set_index(["dataset", "seed", "fold"]).best_epoch
            i = a.index.intersection(b.index)
            if len(i) == 0:
                continue
            dd = (a.loc[i] - b.loc[i]).to_numpy()
            p = stats.wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
            tw.append(dict(freq=f, setting=st, inputs=gs, model=f"{qm}/{rm}", fits=len(i), median=np.median(a.loc[i]),
                           median_real=np.median(b.loc[i]), median_diff=np.median(dd), mean_diff=dd.mean(),
                           p_wilcoxon=p, kind="twin"))
    t = pd.concat([s, pd.DataFrame(tw)], ignore_index=True)
    t.to_csv(OUT_T / "i7_best_epoch.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(s[s.inputs == "PMSE"].drop(columns="kind").round(2).to_string(index=False))
    print(pd.DataFrame(tw).round(3).to_string(index=False))
    print("overall median best epoch:", raw.best_epoch.median(), "| share of fits stopped at max epochs:",
          round(float(np.mean(raw.epochs >= 150)), 4))
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for arg in sys.argv[1:] or ["seed_equiv", "seed_twins", "boot_coins", "econ_level", "hars_inter", "har_market",
                                "lgbm_lags", "best_epoch"]:
        name, _, opt = arg.partition(":")
        globals()[name](*([tuple(opt.split(","))] if opt else []))
