"""
Main comparisons of paper v2 and the checks of review round 2 (docs/paper/review/v2_round2/), computed
from saved forecasts (pooled 4-hour bars, 15 coins, seeds 0-2 unless stated). Tables go to
docs/paper/manuscript_v2/tables/; the figures are drawn from them by experiments/paper_v2_figures.py.

    .venv/bin/python -W ignore experiments/paper_v2_round2.py [name|all]

  main     v2_main.{tex,csv}   the proposed model against benchmarks and other networks: log-range MSE, QLIKE
                               and the tick loss of the 95% limit relative to HAR; the proposed model against
                               every row (ratio, coins, 90% interval, Holm over the rows)
  robust   v2_robust.{tex,csv} the calendar-aware models on hourly, 4-hourly and daily bars
  twins    v2_twins_all.csv    every quaternion-against-real test of the paper in one family (Holm over all
                               rows; Fig. "twins")
  depth    v2_depth.{tex,csv}  MSE relative to HAR of the real and quaternion encoders by depth (run twins first)
  scope    v2_scope.csv        recurrent twins against the parameter budget and the learning rate (Fig. "scope")
  seeds    v2_seeds10.csv      pairs with five or ten seeds: ratio of the averages of all seeds, and of single
                               networks and averages of three and five with the variation between seeds
  bench    v2_bench.csv        the proposed model against benchmarks and the most accurate real models, with
                               the ratio by test block
  benchx   v2_benchx.csv       variants of the linear benchmarks (benchx_save fits and saves them first)
  width    v2_equal_width.csv  quaternion model at 20,000 parameters against the real model of about the same
                               state width at 80,000 parameters
  small    v2_small_checks.csv single contrasts: nonlinear fusions, pooled against single-coin training,
                               single networks against single LightGBM-S fits, budgets
  qlike    v2_qlike_bar.csv    calibrated QLIKE with and without the bar of 10 October 2025 and with the two
                               level constants; share of the summed loss in the largest terms
  blocks   v2_blocks.csv       MSE relative to HAR by test block and for the top decile of the realised range
  select   v2_selection.csv    validation MSE of the first fold (before every test block) with and without
                               the calendar term
  levels   v2_levels.csv       mean MSE of HAR and variance of the target per bar length
  exceed   v2_exceed.csv       95% range limits with a constant set on the validation block: exceedance rate,
                               mean limit and tick loss on the test blocks
  returns  v2_returns_pooled.csv  negative control with pooled training: next-bar log return
  mcs      v2_mcs.csv          model confidence set of the models of the main table
  lrselect v2_lrselect.csv     twin ratios with the learning rate chosen per model and fold on the validation block
  rangein  v2_rangein.csv      first-stage control: the log range of the bar in place of the zero real part

Forecasts: the v2 runs (results/v2*), and the benchmarks and one-layer controls of the first stage of the
study restored from the Drive archive (results/final_*, r8_local, r11_local).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from src.paper.evaluate import _to_time, diebold_mariano, dm_cross, dm_table, holm, load_run  # noqa: E402
import paper_v2_summary as S  # noqa: E402
import paper_v2_tables as T  # noqa: E402

OUT = T.OUT
HAR, PROPOSED = S.HAR, T.PROPOSED
TWIN = ("PMSEC", "hl_meta-L2+cal-pool")
# Forecasts are stamped with the bar at whose close they are made. The target bar that opens at 20:00 UTC on
# 10 October 2025 holds an ATOM low print of 0.001 against a high of 3.94 (log range 8.3).
CRASH_BAR = pd.Timestamp("2025-10-10 16:00", tz="UTC")


def _tag(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Mark the models of a control run (other budget or learning rate) so that they do not collide."""
    return df.assign(model=df.model + tag)


def parts_4h() -> list[pd.DataFrame]:
    """The forecasts behind the main table plus the control runs on 4-hour bars (v2 runs and the controls of
    the earlier stage of the study, all on the same data snapshot, folds and training protocol)."""
    return [T.load(),
            S.run("results/v2g/out/v2_fusion_pool_4h_*"), S.run("results/v2g/out/v2_sources_pool_4h_*"),
            S.run("results/r8_local/final_harsx_4h_r8"), S.run("results/r11_local/final_harxsx_4h_r11"),
            S.run("results/final_tm/out/final_twinmatch_pool_4h_*"),
            S.run("results/r8_local/final_lgbmlags_4h_r8"),
            _tag(S.run("results/v2/out/v2_depth_80000_pool_4h_*"), "@80k"),
            _tag(S.run("results/v2ctl/out/v2_cnn_lr3e4_pool_4h_*"), "@lr"),
            # earlier stage: one-layer models at other budgets and learning rates, the projected real twin of
            # the one-layer hierarchical pair, and random partitions of the 16 features into four groups
            _tag(S.run("results/final_fz/out/final_budget_2000_pool_4h_*"), "@2k1"),
            _tag(S.run("results/final_fz/out/final_budget_5000_pool_4h_*"), "@5k1"),
            _tag(S.run("results/final_fz/out/final_budget_80000_pool_4h_*"), "@80k1"),
            _tag(S.run("results/final_r7a/out/final_lr3e3_pool_4h_*"), "@lr3e-3"),
            _tag(S.run("results/final_r7a/out/final_lr3e4_pool_4h_*"), "@lr3e-4"),
            S.run("results/final_dr2/out/final_grouproj_pool_4h_*"),
            S.run("results/final_rc/out/final_randgroups_pool_4h_*")] + (
        [S.run(BENCHX_DIR)] if os.path.isdir(BENCHX_DIR) else []) + (
        [S.run(LGBM_LAG16_DIR)] if os.path.isdir(LGBM_LAG16_DIR) else []) + (
        [S.run("results/v2local/v2_carr_4h_*")] if glob.glob("results/v2local/v2_carr_4h_*") else []) + [
        _tag(_complete(S.run(pattern)), tag) for pattern, tag in ROUND2_RUNS if glob.glob(pattern)]


# runs added for review round 2 (seeds 0-2); a model enters only when all its cells are present
ROUND2_RUNS = [("results/v2i/out/v2_tfmatch_cal_pool_4h_*", ""), ("results/v2k/out/v2_tfmatch_pool_4h_*", "@m"),
               ("results/v2i/out/v2_lr3e3_cal_pool_4h_*", "@lr3e-3"),
               ("results/v2i/out/v2_lr3e3_sources_pool_4h_*", "@lr3e-3"),
               ("results/v2j/out/v2_budget_2000_cal_pool_4h_*", "@2k"), ("results/v2j/out/v2_budget_5000_cal_pool_4h_*", "@5k"),
               ("results/v2j/out/v2_budget_80000_cal_pool_4h_*", "@80k")]


def _complete(df: pd.DataFrame, cells: int = S.N_CELLS) -> pd.DataFrame:
    n = df.groupby(["groupset", "model"]).apply(lambda g: len(g[["dataset", "seed", "fold"]].drop_duplicates()), include_groups=False)
    short = set(n[n < cells].index)
    if short:
        print("incomplete, left out:", sorted(short))
    return df[[(g, m) not in short for g, m in zip(df.groupset, df.model)]]


class Store:
    """Forecasts split by (groupset, model), so that a pair is selected without scanning one large frame."""

    def __init__(self, parts: list[pd.DataFrame]):
        self.by = {}
        for df in parts:
            assert not df.duplicated(["dataset", "groupset", "model", "seed", "fold", "time"]).any()
            for k, g in df.groupby(["groupset", "model"]):
                assert k not in self.by, f"two runs supply {k}"
                self.by[k] = g
        if HAR in self.by:          # every run must hold the same target on the same bar (same data and folds)
            ref = self.by[HAR].set_index(["dataset", "time"]).y
            for k, g in self.by.items():
                one = g[g.seed == g.seed.min()].set_index(["dataset", "time"]).y
                both = one.index.intersection(ref.index)
                assert len(both) > 0.9 * len(one), f"{k}: bars do not match those of HAR"
                gap = (one.loc[both] - ref.loc[both]).abs().max()
                assert gap < 1e-4, f"{k}: targets differ from those of HAR by up to {gap}"

    def get(self, *keys) -> pd.DataFrame:
        missing = [k for k in keys if k not in self.by]
        if missing:
            raise KeyError(f"no forecasts for {missing}")
        return pd.concat([self.by[k] for k in keys], ignore_index=True)

    def has(self, *keys) -> bool:
        return all(k in self.by for k in keys)

    def vs_har(self, key: tuple) -> float:
        """MSE over all coins and test bars relative to that of HAR."""
        return 1.0 if key == HAR else S.pooled_ratio(self.get(key, HAR), key, HAR)[0]

    def pair(self, a: tuple, b: tuple, margin: float = S.MARGIN) -> dict:
        """a against b: ratio of the MSE over all coins and test bars (`ratio`, also kept as `pooled`) with its
        90% interval, coins where a is better, cross-coin DM test, equivalence at `margin`. `coin_mean` is the
        mean over coins of the per-coin ratios, for reference only."""
        x = self.get(a, b)
        t = dm_table(x, {"range": b})
        c = dm_cross(x, a, b)
        r, lo, hi = S.pooled_ratio(x, a, b)
        return dict(ratio=r, better=int((t.rel_loss < 1).sum()), n=len(t), dm=c["dm_stat"], p=c["p"],
                    pooled=r, ci_lo=lo, ci_hi=hi, equivalent=bool(lo > 1 - margin and hi < 1 + margin),
                    coin_mean=t.rel_loss.mean())

    def by_block(self, a: tuple, b: tuple) -> list[float]:
        """MSE(a) / MSE(b) over all coins within each test block."""
        x = self.get(a, b)
        return [S.pooled_ratio(x[x.fold == k], a, b)[0] for k in sorted(x.fold.unique())]


# ============================================================================================ bench
BENCH_ROWS = [
    ("HAR", HAR), ("HAR-S", ("PMSE", "hars")), ("HAR-XS", ("PMSE", "harxs")),
    ("HAR-S, week cells", ("P", "hars_x")), ("HAR-XS, week cells", ("PMSE", "harxs_x")),
    ("LightGBM-S, pooled", ("PMSE", "lgbms-pool")), ("LightGBM-S, pooled, exchange features", ("PM", "lgbms-pool")),
    ("LightGBM-S, pooled, lagged exchange features", ("PM", "lgbmsl-pool")),
    ("LSTM + calendar", ("PMSEC", "lstm+cal-pool")), ("Transformer + calendar", ("PMSEC", "transformer+cal-pool")),
    ("H-LSTM + calendar (real twin)", TWIN),
    ("H-LSTM, concatenation, four sources", ("PMSEC", "hl_concat-L2+cal-pool")),
    ("H-LSTM, concatenation, price and market", ("PMC", "hl_concat-L2+cal-pool")),
    ("H-QLSTM, concatenation, price and market", ("PMC", "hq_concat-L2+cal-pool")),
    ("H-LSTM, attention fusion, calendar as a group (first design)", ("PMC", "hl_gattn-pool")),
]


def bench(st: Store) -> pd.DataFrame:
    rows = []
    for label, key in BENCH_ROWS:
        r = dict(model=label, mse_har=st.vs_har(key))
        r.update({f"prop_{k}": v for k, v in st.pair(PROPOSED, key).items()})
        for k, v in enumerate(st.by_block(PROPOSED, key)):
            r[f"block{k + 1}"] = v
        rows.append(r)
    t = pd.DataFrame(rows)
    t["prop_p_holm"] = holm(t.prop_p)
    t.to_csv(OUT / "v2_bench.csv", index=False)
    return t


# ============================================================================================ benchx
BENCHX_DIR = "results/v2local/v2_benchx_4h"
LONG = 180                      # bars of the long HAR horizon added as a sensitivity (30 days of 4-hour bars)


def benchx_save():
    """Variants of the linear benchmarks, single-coin OLS on each training block, saved in the runner format:
      PM / harxs, PM / harxs_x      HAR-XS with the eight exchange features only (additive dummies, week cells)
      P / har_l, P / hars_xl, PMSE / harxs_xl   HAR, HAR-S and HAR-XS with week cells plus a 180-bar average
      P / hars_xv, PMSE / harxs_xv  the week-cell models estimated on the training and the validation block
      PM / harxs_xlag               HAR-XS with week cells on the eight exchange features and their averages over
                                    the two longer HAR horizons (the regressors of the lagged LightGBM-S)
    """
    import json
    from paper_round7_checks import _frames, _interactions
    from src.paper.baselines import BAR, HAR_LAGS, _calendar, _har_design, _ols
    out = os.path.join(BENCHX_DIR, "preds")
    os.makedirs(out, exist_ok=True)
    json.dump({"config": {"name": "v2_benchx_4h", "note": "paper_v2_round2.benchx_save"}},
              open(os.path.join(BENCHX_DIR, "config.json"), "w"), indent=1)
    fr, win = _frames("4h")
    for ds, (d, folds) in fr.items():
        y = d.frame["y_range"].to_numpy(float)
        lr = d.frame["log_range"].to_numpy(float)
        H = _har_design(lr, HAR_LAGS["4h"])
        long = _har_design(lr, (LONG,))
        week = _interactions(d.frame.index, d.step)
        add = _calendar(d.frame.index, pd.Timedelta(BAR["4h"]))
        pm, pmse = d.X(["price", "market"]).astype(float), d.X(["price", "market", "social", "sentiment"]).astype(float)
        pm_lags = np.column_stack([pd.DataFrame(pm).rolling(k).mean().to_numpy() for k in HAR_LAGS["4h"][1:]])
        designs = {("PM", "harxs", "train"): [H, pm, add], ("PM", "harxs_x", "train"): [H, pm, week],
                   ("P", "har_l", "train"): [H, long], ("P", "hars_xl", "train"): [H, long, week],
                   ("PMSE", "harxs_xl", "train"): [H, long, pmse, week],
                   ("P", "hars_xv", "both"): [H, week], ("PMSE", "harxs_xv", "both"): [H, pmse, week],
                   ("PM", "harxs_xlag", "train"): [H, pm, pm_lags, week]}
        idx = d.frame.index if d.frame.index.tz is None else d.frame.index.tz_convert(None)
        t = idx.as_unit("ns").asi8
        for (gs, name, fit), cols in designs.items():
            D = np.column_stack(cols)
            for k, fo in enumerate(folds):
                rows = fo["train"] if fit == "train" else np.concatenate([fo["train"], fo["val"]])
                ok = rows[~np.isnan(D[rows]).any(1)]
                b = _ols(D[ok], y[ok])
                te, va = (r[r >= win - 1] for r in (fo["test"], fo["val"]))
                np.savez(os.path.join(out, f"{ds}__range__{gs}__{name}__s0__f{k}.npz"),
                         test_time=t[te], test_y=y[te].astype("float32"), test_pred=(b[0] + D[te] @ b[1:]).astype("float32"),
                         val_time=t[va], val_y=y[va].astype("float32"), val_pred=(b[0] + D[va] @ b[1:]).astype("float32"),
                         meta=np.array(json.dumps(dict(freq="4h", fold=k, model=name, fit=fit))))
    print("saved", out)


BENCHX_ROWS = [("HAR", HAR), ("HAR + 30-day average", ("P", "har_l")),
               ("HAR-S, week cells", ("P", "hars_x")), ("HAR-S, week cells + 30-day average", ("P", "hars_xl")),
               ("HAR-S, week cells, fitted on training and validation blocks", ("P", "hars_xv")),
               ("HAR-XS", ("PMSE", "harxs")), ("HAR-XS, exchange features", ("PM", "harxs")),
               ("HAR-XS, week cells", ("PMSE", "harxs_x")), ("HAR-XS, week cells, exchange features", ("PM", "harxs_x")),
               ("HAR-XS, week cells + 30-day average", ("PMSE", "harxs_xl")),
               ("HAR-XS, week cells, fitted on training and validation blocks", ("PMSE", "harxs_xv")),
               ("HAR-XS, week cells, lagged exchange features", ("PM", "harxs_xlag")),
               ("HAR-S, panel", ("PMSE", "hars-pool")), ("HAR-XS, panel", ("PMSE", "harxs-pool")),
               ("CARR(1,1)", ("P", "carr")), ("LightGBM-S, pooled, lagged, all 16 features", ("PMSE", "lgbmsl-pool"))]


def benchx(st: Store) -> pd.DataFrame:
    rows = []
    for label, key in BENCHX_ROWS:
        if not st.has(key):
            print("not available:", label)
            continue
        v = st.pair(PROPOSED, key)
        rows.append(dict(model=label, mse_har=st.vs_har(key), prop_ratio=v["ratio"], prop_better=v["better"], prop_dm=v["dm"],
                         prop_p=v["p"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_benchx.csv", index=False)
    return t


LGBM_LAG16_DIR = "results/v2local/v2_lgbmlags16_4h"


def lgbmlag16_save(seeds=(0, 1, 2)):
    """The lagged pooled LightGBM-S with all 16 features (the benchmark of paper_round8_checks.lgbm_save, which
    uses the eight exchange features): current values and averages over the two longer HAR horizons. Saved as
    PMSE / lgbmsl-pool in the runner format."""
    import json
    import lightgbm  # noqa: F401  (its OpenMP runtime first, as in the runner)
    from paper_round7_checks import _frames
    from src.paper.baselines import HAR_LAGS, predict_baseline_pooled
    out = os.path.join(LGBM_LAG16_DIR, "preds")
    os.makedirs(out, exist_ok=True)
    json.dump({"config": {"name": "v2_lgbmlags16_4h", "note": "paper_v2_round2.lgbmlag16_save"}},
              open(os.path.join(LGBM_LAG16_DIR, "config.json"), "w"), indent=1)
    fr, win = _frames("4h")
    dss = list(fr)
    Xs = [fr[ds][0].X(["price", "market", "social", "sentiment"]) for ds in dss]
    Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS["4h"][1:]]) for X in Xs]
    for seed in seeds:
        for k in range(len(fr[dss[0]][1])):
            res = predict_baseline_pooled("lgbms", [fr[ds][0].frame for ds in dss], "range", [fr[ds][1][k] for ds in dss],
                                          win, "4h", Xl, seed=seed)
            for ds, r in zip(dss, res):
                frame = fr[ds][0].frame
                idx = frame.index if frame.index.tz is None else frame.index.tz_convert(None)
                t = idx.as_unit("ns").asi8
                y = frame["y_range"].to_numpy(float)
                np.savez(os.path.join(out, f"{ds}__range__PMSE__lgbmsl-pool__s{seed}__f{k}.npz"),
                         test_time=t[r["test_rows"]], test_y=y[r["test_rows"]].astype("float32"),
                         test_pred=r["test_pred"].astype("float32"), val_time=t[r["val_rows"]],
                         val_y=y[r["val_rows"]].astype("float32"), val_pred=r["val_pred"].astype("float32"),
                         meta=np.array(json.dumps(dict(freq="4h", seed=seed, fold=k, model="lgbmsl-pool"))))
            print("seed", seed, "fold", k, flush=True)


# ============================================================================================ mcs
def mcs(st: Store, size: float = 0.10) -> pd.DataFrame:
    """Model confidence set (Hansen, Lunde and Nason 2011) of the models of the main table: loss = squared
    error averaged over the coins on each test bar; block bootstrap with blocks of n^(1/3) bars, 2,000
    replications; a model is in the set at level 1 - size if its MCS p-value is at least `size`."""
    from arch.bootstrap import MCS
    keys = [k for _, _, k in MAIN_ROWS if k is not None] + [("P", "carr")]
    keys = [k for k in dict.fromkeys(keys) if st.has(k)]
    cols = {}
    for k in keys:
        e = st.by[k].groupby(["dataset", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
        e["se"] = (e.pred - e.y) ** 2
        cols[k] = e.pivot(index="time", columns="dataset", values="se").dropna().mean(axis=1)
    L = pd.DataFrame(cols).dropna()
    m = MCS(L.to_numpy(), size=size, reps=2000, block_size=max(2, int(len(L) ** (1 / 3))), seed=0)
    m.compute()
    pv = m.pvalues.squeeze()
    t = pd.DataFrame([dict(groupset=k[0], model=k[1], mse=L[k].mean(), mcs_p=float(pv.loc[i]), in_set=bool(pv.loc[i] >= size))
                      for i, k in enumerate(L.columns)]).sort_values("mse")
    t.to_csv(OUT / "v2_mcs.csv", index=False)
    return t


# ============================================================================================ twins
# (group, label, quaternion key, real key) on the 4-hour frame
def _twin_rows_4h() -> list[tuple]:
    rows = [("Pooled, 20,000 parameters" if "calendar" not in fam else "With the calendar term",
             f"{fam.replace(' + calendar', '')}, {depth}", (gs, q + "-pool"), (gs, r + "-pool"))
            for fam, depth, q, r, gs in S.TWIN_ROWS]
    # real Transformer twins with the activation and heads of the quaternion model: the flat one of stage 1 has the
    # size of the standard real Transformer (9% below the quaternion model), the later one is sized to the quaternion model
    rows += [("Matched Transformer twins", "Transformer, 2 layers, activation and heads", ("PMSE", "qtransformer-pool"), ("PMSE", "transformer_g-pool")),
             ("Matched Transformer twins", "Transformer, 2 layers, and size", ("PMSE", "qtransformer-pool"), ("PMSE", "transformer_g-pool@m")),
             ("Matched Transformer twins", "Hierarchical Transformer, 1 layer", ("PMSE", "hqt_meta-pool"), ("PMSE", "htg_meta-pool"))]
    for label, q, r in [("LSTM, 3 layers", "qlstm-L3", "lstm-L3"), ("Hierarchical LSTM, 2 layers", "hq_meta-L2", "hl_meta-L2"),
                        ("CNN, 6 conv.", "qcnn-L6", "cnn-L6"), ("Hierarchical CNN, 4 conv.", "hqc_meta", "hc_meta")]:
        rows.append(("80,000 parameters", label, ("PMSE", f"{q}-pool@80k"), ("PMSE", f"{r}-pool@80k")))
    for label, q, r in [("CNN, 2 conv.", "qcnn-L2", "cnn-L2"), ("CNN, 4 conv.", "qcnn", "cnn"), ("CNN, 6 conv.", "qcnn-L6", "cnn-L6"),
                        ("Hierarchical CNN, 2 conv.", "hqc_meta-L2", "hc_meta-L2"), ("Hierarchical CNN, 4 conv.", "hqc_meta", "hc_meta")]:
        rows.append(("Learning rate 3e-4", label, ("PMSE", f"{q}-pool@lr"), ("PMSE", f"{r}-pool@lr")))
    for label, q, r in SINGLE_AND_DAILY:
        rows.append(("One model per coin", label, ("PMSE", q), ("PMSE", r)))
    for label, f in [("Concatenation", "concat"), ("Dense", "dense"), ("Hamilton + tanh", "metan"), ("Dense + tanh", "densen")]:
        rows.append(("Other fusion operators (2 layers, calendar)", label,
                     ("PMSEC", f"hq_{f}-L2+cal-pool"), ("PMSEC", f"hl_{f}-L2+cal-pool")))
    rows.append(("Other fusion operators (2 layers, calendar)", "Hamilton, projected real twin", PROPOSED,
                 ("PMSEC", "hlp_meta-L2+cal-pool")))
    for label, gs in [("Price, market", "PMC"), ("Price, market, social", "PMSC"), ("Price, market, sentiment", "PMEC")]:
        rows.append(("Fewer sources (concatenation, 2 layers, calendar)", label,
                     (gs, "hq_concat-L2+cal-pool"), (gs, "hl_concat-L2+cal-pool")))
    # runs of review round 2: rows are added when the run is complete (see twins)
    rows += [("Matched Transformer twins, calendar", "Transformer, 2 layers", ("PMSEC", "qtransformer+cal-pool"), ("PMSEC", "transformer_g+cal-pool")),
             ("Matched Transformer twins, calendar", "Hierarchical Transformer, 1 layer", ("PMSEC", "hqt_meta+cal-pool"), ("PMSEC", "htg_meta+cal-pool")),
             ("Proposed pair, other settings", "Learning rate 3e-3", ("PMSEC", "hq_meta-L2+cal-pool@lr3e-3"), ("PMSEC", "hl_meta-L2+cal-pool@lr3e-3"))]
    for name, tag in [("2,000 parameters", "@2k"), ("5,000 parameters", "@5k"), ("80,000 parameters", "@80k")]:
        rows.append(("Proposed pair, other settings", name, ("PMSEC", f"hq_meta-L2+cal-pool{tag}"), ("PMSEC", f"hl_meta-L2+cal-pool{tag}")))
    for label, q, r in [("LSTM, 1 layer", "qlstm", "lstm"), ("Hierarchical LSTM, 1 layer", "hq_meta", "hl_meta")]:
        for name, tag in [("2,000 parameters", "@2k1"), ("5,000 parameters", "@5k1"), ("80,000 parameters", "@80k1")]:
            rows.append(("Other budgets (1 layer)", f"{label}, {name}", ("PMSE", f"{q}-pool{tag}"), ("PMSE", f"{r}-pool{tag}")))
    for label, q, r in [("LSTM, 1 layer", "qlstm", "lstm"), ("Hierarchical LSTM, 1 layer", "hq_meta", "hl_meta"),
                        ("Transformer, 2 layers", "qtransformer", "transformer"),
                        ("Hierarchical Transformer, 1 layer", "hqt_meta", "ht_meta")]:
        for name, tag in [("learning rate 3e-4", "@lr3e-4"), ("learning rate 3e-3", "@lr3e-3")]:
            rows.append(("Other learning rates (1 layer)", f"{label}, {name}", ("PMSE", f"{q}-pool{tag}"), ("PMSE", f"{r}-pool{tag}")))
    rows.append(("Projected real twin (1 layer)", "Hierarchical LSTM, 1 layer", ("PMSE", "hq_meta-pool"), ("PMSE", "hlp_meta-pool")))
    for gs in ("R1", "R2", "R3"):
        rows.append(("Random feature groups (1 layer)", f"Hierarchical LSTM, partition {gs[1]}", (gs, "hq_meta-pool"), (gs, "hl_meta-pool")))
    return rows


SINGLE_AND_DAILY = [("LSTM, 1 layer", "qlstm", "lstm"), ("LSTM, 2 layers", "qlstm-L2", "lstm-L2"),
                    ("Hierarchical LSTM, 1 layer", "hq_meta", "hl_meta"), ("Hierarchical LSTM, 2 layers", "hq_meta-L2", "hl_meta-L2"),
                    ("CNN, 6 conv.", "qcnn-L6", "cnn-L6"), ("Hierarchical CNN, 4 conv.", "hqc_meta", "hc_meta")]


def twins(st: Store) -> pd.DataFrame:
    rows = []
    for group, label, a, b in _twin_rows_4h():
        if not st.has(a, b):
            print(f"not available yet: {group} / {label}")
            continue
        rows.append(dict(group=group, pair=label, bars="4h", **st.pair(a, b)))
    daily = Store([S.run("results/v2ctl/out/v2_depth_pool_1d_*").query("model != 'har'"), S.run("results/v2f/out/v2_cal_pool_1d_*")])
    for label, q, r in SINGLE_AND_DAILY:
        rows.append(dict(group="Daily bars", pair=label, bars="1d", **daily.pair(("PMSE", q + "-pool"), ("PMSE", r + "-pool"))))
    rows.append(dict(group="Daily bars", pair="Hierarchical LSTM, 2 layers, calendar", bars="1d", **daily.pair(PROPOSED, TWIN)))
    hourly = Store([S.run("results/v2f/out/v2_cal_pool_1h_*")])
    rows.append(dict(group="Hourly bars", pair="Hierarchical LSTM, 2 layers, calendar", bars="1h", **hourly.pair(PROPOSED, TWIN)))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    t["half_width"] = (t.ci_hi - t.ci_lo) / 2
    t.to_csv(OUT / "v2_twins_all.csv", index=False)
    return t


# ============================================================================================ width
def width(st: Store) -> pd.DataFrame:
    """Parameter efficiency at an equal state width: the quaternion model at the 20,000 budget against the
    real model at the 80,000 budget (state components per encoder from Table 4)."""
    rows = []
    for label, q, r, comp in [("LSTM, 3 layers", "qlstm-L3", "lstm-L3", "56 / 61"),
                              ("Hierarchical LSTM, 2 layers", "hq_meta-L2", "hl_meta-L2", "36 / 36"),
                              ("CNN, 6 conv.", "qcnn-L6", "cnn-L6", "60 / 62"),
                              ("Hierarchical CNN, 4 conv.", "hqc_meta", "hc_meta", "32 / 30")]:
        a, b = ("PMSE", f"{q}-pool"), ("PMSE", f"{r}-pool@80k")
        rows.append(dict(pair=label, components=comp, **st.pair(a, b)))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    t.to_csv(OUT / "v2_equal_width.csv", index=False)
    return t


# ============================================================================================ seeds
def _cube(df: pd.DataFrame, key: tuple) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Forecasts of one model as an array (seed, coin, bar) and the targets as (coin, bar), on the bars that
    every coin and seed has."""
    g = df[(df.groupset == key[0]) & (df.model == key[1])]
    wide = g.pivot_table(index="time", columns=["seed", "dataset"], values="pred").dropna()
    y = g[g.seed == g.seed.min()].pivot(index="time", columns="dataset", values="y").loc[wide.index]
    seeds, coins = sorted(g.seed.unique()), list(y.columns)
    P = np.stack([wide[s][coins].to_numpy().T for s in seeds])
    return P, y.to_numpy().T, wide.index


def _all_seeds(patterns: list, keys: list[tuple]) -> pd.DataFrame:
    """Forecasts of `keys` from the runs in `patterns`. An entry (pattern, tag) marks the models of that run
    with the tag (a run at another budget uses the model names of the main run)."""
    parts = []
    for entry in patterns:
        pattern, tag = (entry, "") if isinstance(entry, str) else entry
        for d in glob.glob(pattern):
            x = _tag(load_run(d), tag)
            parts.append(x[(x.task == "range").to_numpy() & np.array([(g, m) in keys for g, m in zip(x.groupset, x.model)])])
    df = pd.concat(parts, ignore_index=True)
    assert not df.duplicated(["dataset", "groupset", "model", "seed", "fold", "time"]).any()
    return df


def seed_time(Pa: np.ndarray, Pb: np.ndarray, Y: np.ndarray, k: int, draws: int = 4000, block: int = 42,
              seed: int = 0) -> dict:
    """Ratio of the coin-averaged MSE of an average of k networks of model a to that of k networks of model b.
    `seeds`: the k networks of each arm are drawn without replacement from those trained, the test bars are
    kept (variation between seeds only). `joint`: in addition the test bars are resampled in blocks of
    `block` bars (moving-block bootstrap), which gives an interval over seeds and time."""
    rng = np.random.default_rng(seed)
    n = Y.shape[1]
    only, joint = np.empty(draws), np.empty(draws)
    for i in range(draws):
        a = rng.choice(Pa.shape[0], k, replace=False)
        b = rng.choice(Pb.shape[0], k, replace=False)
        la = ((Pa[a].mean(0) - Y) ** 2).mean(0)
        lb = ((Pb[b].mean(0) - Y) ** 2).mean(0)
        only[i] = la.mean() / lb.mean()
        starts = rng.integers(0, n - block + 1, size=-(-n // block))
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        joint[i] = la[idx].mean() / lb[idx].mean()
    q = lambda v, p: float(np.quantile(v, p))
    return dict(k=k, mean=float(only.mean()), sd_seeds=float(only.std(ddof=1)), seeds_lo=q(only, 0.05), seeds_hi=q(only, 0.95),
                share_inside_1pct=float(((only > 0.99) & (only < 1.01)).mean()),
                joint_lo=q(joint, 0.05), joint_hi=q(joint, 0.95), sd_joint=float(joint.std(ddof=1)))


RUNS_20K = ["results/v2cal/out/v2_cal_pool_4h_*", "results/v2h/out/v2_seeds_pool_4h_*"]
# the proposed pair at 2,000 parameters: seeds 0-2 on a GPU; seeds 3-9 of the quaternion model on a CPU and of
# the real twin on a GPU (configs/paper/v2/seeds_budget_2000_cal_pool_4h.yaml and ..._twin.yaml)
RUNS_2K = [("results/v2j/out/v2_budget_2000_cal_pool_4h_*", "@2k"), ("results/v2local/v2_seeds_budget_2000_cal_pool_4h_*", "@2k"),
           ("results/v2s/out/v2_seeds_budget_2000_cal_pool_4h_twin_*", "@2k")]
PROPOSED_2K, TWIN_2K = (PROPOSED[0], PROPOSED[1] + "@2k"), (TWIN[0], TWIN[1] + "@2k")
RUNS_LGBM = ["results/final_local/final_lgbm_4h_*", "results/v2local/v2_lgbm_seeds_4h_*", "results/r8_local/final_lgbmlags_4h_r8"]

SEED_PAIRS = [   # (label, quaternion or first key, real or second key, run patterns)
    ("H-QLSTM / H-LSTM, 2 layers, calendar (proposed pair)", PROPOSED, TWIN, RUNS_20K),
    ("Proposed pair at 2,000 parameters", PROPOSED_2K, TWIN_2K, RUNS_2K),
    ("H-QLSTM, 2,000 / 20,000 parameters", PROPOSED_2K, PROPOSED, RUNS_2K + RUNS_20K),
    ("H-LSTM, 2,000 / 20,000 parameters", TWIN_2K, TWIN, RUNS_2K + RUNS_20K),
    ("H-QLSTM at 2,000 / H-LSTM at 20,000 parameters", PROPOSED_2K, TWIN, RUNS_2K + RUNS_20K),
    ("Proposed, all seeds / LightGBM-S", PROPOSED, ("PMSE", "lgbms-pool"), RUNS_20K + RUNS_LGBM),
    ("Proposed, all seeds / LightGBM-S, lagged", PROPOSED, ("PM", "lgbmsl-pool"), RUNS_20K + RUNS_LGBM),
    ("Proposed / H-LSTM on price and market", PROPOSED, ("PMC", "hl_concat-L2+cal-pool"),
     ["results/v2cal/out/v2_cal_pool_4h_*", "results/v2h/out/v2_seeds_pool_4h_*", "results/v2g/out/v2_sources_pool_4h_*",
      "results/v2i/out/v2_seeds_sources_pool_4h_*"]),
    ("H-QCNN / H-CNN, 2 convolutions", ("PMSE", "hqc_meta-L2-pool"), ("PMSE", "hc_meta-L2-pool"),
     ["results/v2/out/v2_depth_pool_4h_*", "results/v2i/out/v2_seeds_hcnn_pool_4h_*"]),
    ("QLSTM / LSTM, 1 layer (earlier run)", ("PMSE", "qlstm-pool"), ("PMSE", "lstm-pool"), ["results/final_fa/out/final_pool_4h_*"]),
    ("H-QLSTM / H-LSTM, 1 layer (earlier run)", ("PMSE", "hq_meta-pool"), ("PMSE", "hl_meta-pool"), ["results/final_fa/out/final_pool_4h_*"]),
    ("QTransformer / Transformer (earlier run)", ("PMSE", "qtransformer-pool"), ("PMSE", "transformer-pool"), ["results/final_fa/out/final_pool_4h_*"]),
    ("H-QTransformer / H-Transformer (earlier run)", ("PMSE", "hqt_meta-pool"), ("PMSE", "ht_meta-pool"), ["results/final_fa/out/final_pool_4h_*"]),
]


def seeds() -> pd.DataFrame:
    """Twin ratios with all trained seeds: the ratio for averages of all seeds with its interval over time, and
    for single networks and averages of three and five networks with the variation between seeds included."""
    har = S.run("results/final_fa/out/final_single_4h_*")
    har = har[(har.model == "har") & (har.groupset == "P")]
    rows = []
    for label, a, b, patterns in SEED_PAIRS:
        if not all(glob.glob(pt if isinstance(pt, str) else pt[0]) for pt in patterns):
            print(f"[{label}] runs missing, skipped")
            continue
        df = _all_seeds(patterns, [a, b]).drop_duplicates(["dataset", "groupset", "model", "seed", "fold", "time"], keep="last")
        n = df.groupby(["groupset", "model", "seed"]).apply(lambda g: len(g[["dataset", "fold"]].drop_duplicates()), include_groups=False)
        full = n[n == 60].reset_index()                       # 15 coins x 4 folds
        df = df.merge(full[["groupset", "model", "seed"]], on=["groupset", "model", "seed"])
        Pa, Y, ta = _cube(df, a)
        Pb, Yb, tb = _cube(df, b)
        assert ta.equals(tb) and np.allclose(Y, Yb, atol=1e-5)
        st = Store([df, har])
        full_pair = st.pair(a, b)
        r = dict(pair=label, seeds_a=Pa.shape[0], seeds_b=Pb.shape[0], mse_har_a=st.vs_har(a), mse_har_b=st.vs_har(b),
                 ratio_all=full_pair["ratio"], better=full_pair["better"], dm=full_pair["dm"], p=full_pair["p"],
                 ci_lo=full_pair["ci_lo"], ci_hi=full_pair["ci_hi"])
        har_mse = ((_har_cube(har, ta, df) - Y) ** 2).mean()                     # over all coins and bars
        single = lambda P: [float(((P[i] - Y) ** 2).mean() / har_mse) for i in range(P.shape[0])]
        sa, sb = single(Pa), single(Pb)
        r.update(single_a_min=min(sa), single_a_max=max(sa), single_b_min=min(sb), single_b_max=max(sb))
        for k in (1, 3, 5):
            if k <= min(Pa.shape[0], Pb.shape[0]) and (k < min(Pa.shape[0], Pb.shape[0]) or k == 1):
                v = seed_time(Pa, Pb, Y, k)
                r.update({f"k{k}_{name}": val for name, val in v.items() if name != "k"})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_seeds10.csv", index=False)
    print(t.set_index("pair").T.round(4).to_string())
    return t[["pair", "seeds_a", "seeds_b", "ratio_all"]]


def _har_cube(har: pd.DataFrame, times: pd.DatetimeIndex, df: pd.DataFrame) -> np.ndarray:
    coins = sorted(df.dataset.unique())
    return har.pivot(index="time", columns="dataset", values="pred").loc[times, coins].to_numpy().T


# ============================================================================================ small checks
def small(st: Store) -> pd.DataFrame:
    """Contrasts that review round 2 asked to be tested: the nonlinear Hamilton fusion against the nonlinear
    dense fusion, pooled against single-coin training, single networks against single LightGBM-S fits, the
    two-layer model against a fourfold budget."""
    rows = []
    add = lambda label, a, b: rows.append(dict(contrast=label, **st.pair(a, b)))
    for enc, name in (("hq", "quaternion encoders"), ("hl", "real encoders")):
        add(f"Hamilton + tanh / dense + tanh, {name}", ("PMSEC", f"{enc}_metan-L2+cal-pool"), ("PMSEC", f"{enc}_densen-L2+cal-pool"))
    for m, label in (("hq_meta-L2", "H-QLSTM, 2 layers"), ("hq_meta", "H-QLSTM, 1 layer"), ("qlstm", "QLSTM, 1 layer"),
                     ("hl_meta-L2", "H-LSTM, 2 layers"), ("hl_meta", "H-LSTM, 1 layer"), ("lstm", "LSTM, 1 layer")):
        add(f"pooled / one model per coin, {label}", ("PMSE", m + "-pool"), ("PMSE", m))
    for m, label in (("hq_meta-L2", "H-QLSTM"), ("hl_meta-L2", "H-LSTM")):
        add(f"20,000 / 80,000 parameters, {label}, 2 layers", ("PMSE", f"{m}-pool"), ("PMSE", f"{m}-pool@80k"))
    add("2,000 parameters H-QLSTM / 20,000 parameters H-LSTM, 1 layer", ("PMSE", "hq_meta-pool@2k1"), ("PMSE", "hl_meta-pool"))
    add("first design: H-QLSTM / H-LSTM, attention fusion, calendar group", ("PMC", "hq_gattn-pool"), ("PMC", "hl_gattn-pool"))
    add("LightGBM-S, exchange features / all 16 features", ("PM", "lgbms-pool"), ("PMSE", "lgbms-pool"))
    add("LightGBM-S, lagged exchange features / same-bar features", ("PM", "lgbmsl-pool"), ("PM", "lgbms-pool"))
    add("H-LSTM first design / LightGBM-S, lagged exchange features", ("PMC", "hl_gattn-pool"), ("PM", "lgbmsl-pool"))
    add("H-LSTM on price and market / LightGBM-S, lagged exchange features", ("PMC", "hl_concat-L2+cal-pool"), ("PM", "lgbmsl-pool"))
    add("QTransformer + calendar / proposed", ("PMSEC", "qtransformer+cal-pool"), PROPOSED)
    # the proposed pair across sizes (three seeds): equal width, and each model against a larger one
    q, r = "hq_meta-L2+cal-pool", "hl_meta-L2+cal-pool"
    for label, a, b in [("H-QLSTM at 2,000 / H-LSTM at 5,000 (equal width), calendar", q + "@2k", r + "@5k"),
                        ("H-QLSTM at 2,000 / H-LSTM at 20,000, calendar", q + "@2k", r),
                        ("H-QLSTM at 2,000 / H-QLSTM at 20,000, calendar", q + "@2k", q),
                        ("H-LSTM at 2,000 / H-LSTM at 20,000, calendar", r + "@2k", r),
                        ("H-QLSTM at 5,000 / H-QLSTM at 20,000, calendar", q + "@5k", q),
                        ("H-LSTM at 5,000 / H-LSTM at 20,000, calendar", r + "@5k", r),
                        ("H-QLSTM at 20,000 / H-QLSTM at 80,000, calendar", q, q + "@80k"),
                        ("H-QLSTM at 20,000 / H-LSTM at 80,000 (equal width), calendar", q, r + "@80k"),
                        ("H-LSTM at 20,000 / H-LSTM at 80,000, calendar", r, r + "@80k")]:
        if st.has(("PMSEC", a), ("PMSEC", b)):
            add(label, ("PMSEC", a), ("PMSEC", b))
    for m, label in (("lstm-L3", "LSTM, 3 layers"), ("qlstm-L3", "QLSTM, 3 layers"), ("cnn-L6", "CNN, 6 conv."),
                     ("qcnn-L6", "QCNN, 6 conv."), ("hc_meta", "H-CNN, 4 conv."), ("hqc_meta", "H-QCNN, 4 conv.")):
        add(f"20,000 / 80,000 parameters, {label}", ("PMSE", f"{m}-pool"), ("PMSE", f"{m}-pool@80k"))
    t = pd.DataFrame(rows)
    # single networks against single LightGBM-S fits, seed by seed
    for sd in (0, 1, 2):
        x = st.get(PROPOSED, ("PMSE", "lgbms-pool"))
        x = x[x.seed == sd]
        one = Store([x])
        t = pd.concat([t, pd.DataFrame([dict(contrast=f"proposed / LightGBM-S, seed {sd}", **one.pair(PROPOSED, ("PMSE", "lgbms-pool")))])],
                      ignore_index=True)
    t.to_csv(OUT / "v2_small_checks.csv", index=False)
    return t


def returns_pooled() -> pd.DataFrame:
    """Negative control with pooled training (earlier stage): next-bar log return, MSE relative to a zero forecast."""
    (d,) = glob.glob("results/final_cr/out/final_return_pool_4h_*")
    df = load_run(d)
    df = df[(df.task == "return") & (df.seed <= 2)]
    ens = df.groupby(["dataset", "groupset", "model", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
    ens["se"], ens["se0"] = (ens.pred - ens.y) ** 2, ens.y ** 2
    g = ens.groupby(["groupset", "model", "dataset"]).agg(mse=("se", "mean"), mse0=("se0", "mean"))
    g["rel"] = g.mse / g.mse0
    t = g.groupby(["groupset", "model"]).agg(mse=("mse", "mean"), mse0=("mse0", "mean"), coins_better=("rel", lambda v: int((v < 1).sum())),
                                              n=("rel", "size")).reset_index()
    t.insert(2, "mse_over_zero", t.pop("mse") / t.pop("mse0"))       # losses averaged over the coins, then the ratio
    t.to_csv(OUT / "v2_returns_pooled.csv", index=False)
    return t


# ============================================================================================ scope
def scope(st: Store) -> pd.DataFrame:
    """MSE relative to HAR of the recurrent twins against the parameter budget and the learning rate: the
    one-layer flat and hierarchical pairs of the earlier stage (no calendar term) and the proposed pair."""
    rows = []

    def add(panel, pair, algebra, x, key):
        if st.has(key):
            rows.append(dict(panel=panel, pair=pair, algebra=algebra, x=x, mse_har=st.vs_har(key)))

    for fam, q, r in [("LSTM, 1 layer", "qlstm", "lstm"), ("H-LSTM, 1 layer", "hq_meta", "hl_meta")]:
        for algebra, m in (("quaternion", q), ("real", r)):
            for x, tag in [(2000, "@2k1"), (5000, "@5k1"), (20000, ""), (80000, "@80k1")]:
                add("budget", fam, algebra, x, ("PMSE", f"{m}-pool{tag}"))
            for x, tag in [(3e-4, "@lr3e-4"), (1e-3, ""), (3e-3, "@lr3e-3")]:
                add("learning rate", fam, algebra, x, ("PMSE", f"{m}-pool{tag}"))
    for algebra, m in (("quaternion", "hq_meta-L2+cal"), ("real", "hl_meta-L2+cal")):
        for x, tag in [(2000, "@2k"), (5000, "@5k"), (20000, ""), (80000, "@80k")]:
            add("budget", "Proposed pair", algebra, x, ("PMSEC", f"{m}-pool{tag}"))
        for x, tag in [(1e-3, ""), (3e-3, "@lr3e-3")]:
            add("learning rate", "Proposed pair", algebra, x, ("PMSEC", f"{m}-pool{tag}"))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_scope.csv", index=False)
    return t


# ============================================================================================ blocks
BLOCK_ROWS = [("HAR-S", ("PMSE", "hars")), ("HAR-S, week cells", ("P", "hars_x")), ("LightGBM-S, pooled", ("PMSE", "lgbms-pool")),
              ("LightGBM-S, pooled, lagged exchange features", ("PM", "lgbmsl-pool")),
              ("H-LSTM, price and market", ("PMC", "hl_concat-L2+cal-pool")), ("H-LSTM (real twin)", TWIN),
              ("H-QLSTM (proposed)", PROPOSED)]


def _rows_ratio(x: pd.DataFrame, key: tuple, ref: tuple, keep=None) -> float:
    """MSE of `key` divided by that of `ref` over the coin-bars selected by `keep` (seed-averaged forecasts)."""
    e = x.groupby(["dataset", "groupset", "model", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
    w = e.pivot_table(index=["dataset", "time"], columns=["groupset", "model"], values="pred")
    w.columns = [tuple(c) for c in w.columns]
    w["y"] = e.groupby(["dataset", "time"]).y.first()
    w = w.dropna()
    if keep is not None:
        w = w[keep(w)]
    return float(((w[key] - w.y) ** 2).mean() / ((w[ref] - w.y) ** 2).mean())


def blocks(st: Store) -> pd.DataFrame:
    """MSE relative to HAR by test block, and on the bars in the top decile of the realised range of each
    coin against the other bars."""
    rows = []
    for label, key in BLOCK_ROWS:
        x = st.get(key, HAR)
        r = dict(model=label)
        for k in sorted(x.fold.unique()):
            r[f"block{k + 1}"] = _rows_ratio(x[x.fold == k], key, HAR)
        top = lambda w: w.y >= w.groupby(level="dataset").y.transform(lambda v: v.quantile(0.9))
        r["top_decile"] = _rows_ratio(x, key, HAR, top)
        r["other_bars"] = _rows_ratio(x, key, HAR, lambda w: ~top(w))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_blocks.csv", index=False)
    return t


# ============================================================================================ qlike
def _splits(keys: list[tuple], patterns: list[str]) -> pd.DataFrame:
    """Seed-averaged validation and test forecasts of `keys` (seeds 0-2). A later pattern overrides an
    earlier one for the same cell."""
    files = {}
    for pattern in patterns:
        for directory in glob.glob(pattern):
            for path in glob.glob(os.path.join(directory, "preds", "*.npz")):
                ds, task, gs, model, seed, fold = os.path.basename(path)[:-4].split("__")
                if task == "range" and (gs, model) in keys and int(seed[1:]) <= 2:
                    files[(ds, gs, model, seed, fold)] = path
    parts = []
    for (ds, gs, model, seed, fold), path in files.items():
        z = np.load(path)
        for split in ("test", "val"):
            parts.append(pd.DataFrame({"dataset": ds, "groupset": gs, "model": model, "seed": int(seed[1:]),
                                       "fold": int(fold[1:]), "split": split, "time": _to_time(z[f"{split}_time"]),
                                       "y": z[f"{split}_y"].astype(float), "pred": z[f"{split}_pred"].astype(float)}))
    df = pd.concat(parts, ignore_index=True)
    missing = set(keys) - set(zip(df.groupset, df.model))
    assert not missing, f"no forecasts for {missing}"
    return (df.groupby(["dataset", "groupset", "model", "fold", "split", "time"], as_index=False)
              .agg(y=("y", "first"), pred=("pred", "mean")))


QLIKE_ROWS = [("HAR", HAR), ("HAR-S", ("PMSE", "hars")), ("HAR-XS", ("PMSE", "harxs")),
              ("HAR-S, week cells", ("P", "hars_x")), ("HAR-XS, week cells", ("PMSE", "harxs_x")),
              ("LightGBM-S, pooled", ("PMSE", "lgbms-pool")), ("LightGBM-S, pooled, lagged exchange features", ("PM", "lgbmsl-pool")),
              ("CNN, 6 convolutions", ("PMSE", "cnn-L6-pool")), ("LSTM", ("PMSE", "lstm-pool")),
              ("Transformer", ("PMSE", "transformer-pool")), ("LSTM + calendar", ("PMSEC", "lstm+cal-pool")),
              ("Transformer + calendar", ("PMSEC", "transformer+cal-pool")),
              ("H-LSTM, 2 layers", ("PMSE", "hl_meta-L2-pool")), ("H-QLSTM, 2 layers", ("PMSE", "hq_meta-L2-pool")),
              ("H-LSTM, price and market, calendar", ("PMC", "hl_concat-L2+cal-pool")),
              ("H-LSTM, first design", ("PMC", "hl_gattn-pool")),
              ("H-LSTM + calendar (real twin)", TWIN), ("H-QLSTM + calendar (proposed)", PROPOSED)]
QLIKE_RUNS = T.MAIN_RUNS + ["results/v2g/out/v2_sources_pool_4h_*", "results/r8_local/final_harsx_4h_r8",
                            "results/r11_local/final_harxsx_4h_r11", "results/r8_local/final_lgbmlags_4h_r8",
                            "results/final_fc/out/final_calendar_pool_4h_*", "results/v2local/v2_carr_4h_*"]


def train_caps(freq: str = "4h") -> pd.Series:
    """Largest realised squared range in the training block of every coin and fold. Used as a cap on the
    realised squared range in QLIKE: a rule that needs no information from the validation or test blocks."""
    from paper_round7_checks import _frames
    fr, _ = _frames(freq)
    rows = []
    for ds, (d, folds) in fr.items():
        y = d.frame["y_range"].to_numpy(float)
        for k, fo in enumerate(folds):
            rows.append(dict(dataset=ds, fold=k, cap=float(np.exp(2 * np.nanmax(y[fo["train"]])))))
    return pd.DataFrame(rows).set_index(["dataset", "fold"]).cap


def _calibrated(ens: pd.DataFrame, drop: pd.Timestamp | None, rule: str = "robust", caps: pd.Series | None = None) -> pd.DataFrame:
    """Test-block QLIKE of the variance forecast k * exp(2 * forecast). The level constant k comes from the
    validation block of the coin and fold:
      "means"   ratio of the mean squared range to the mean raw forecast (the rule of the first draft); one
                extreme range or one extreme forecast in the block moves it without bound;
      "robust"  mean of the ratio squared range / raw forecast, the constant that minimises QLIKE on the block,
                with the ratio winsorised at its 99th percentile, so that no single bar can move it.
    With `drop`, the forecast made at that time is left out of the test loss (and of the block for "means").
    With `caps` (train_caps), the realised squared range is capped at the largest value of the coin's training
    block of that fold, in the validation block and in the test loss."""
    if drop is not None:
        ens = ens[ens.time != drop]
    ens = ens.assign(s2=np.exp(2 * ens.y), raw=np.exp(2 * ens.pred))
    if caps is not None:
        ens = ens.join(caps, on=["dataset", "fold"])
        ens["capped"] = ens.s2 > ens.cap
        ens["s2"] = np.minimum(ens.s2, ens.cap)
    by = ["dataset", "groupset", "model", "fold"]
    val = ens[ens.split == "val"]
    if rule == "means":
        v = val.groupby(by).agg(s2=("s2", "mean"), raw=("raw", "mean"))
        k = (v.s2 / v.raw).rename("k")
    else:
        u = val.assign(u=val.s2 / val.raw)
        cap = u.groupby(by).u.transform(lambda x: x.quantile(0.99))
        k = u.assign(u=np.minimum(u.u, cap)).groupby(by).u.mean().rename("k")
    test = ens[ens.split == "test"].join(k, on=by)
    test["h"] = test.k * test.raw
    test["ql"] = test.s2 / test.h - np.log(test.s2 / test.h) - 1
    return test


def qlike() -> pd.DataFrame:
    ens = _splits([k for _, k in QLIKE_ROWS], QLIKE_RUNS)
    out = {"first draft": _calibrated(ens, None, "means"), "all bars": _calibrated(ens, None),
           "without the bar": _calibrated(ens, CRASH_BAR), "capped": _calibrated(ens, None, caps=train_caps())}
    c = out["capped"]
    c = c[(c.groupset == PROPOSED[0]) & (c.model == PROPOSED[1])]
    print(f"capped rule: {int(c.capped.sum())} of {len(c)} test coin-bars are capped, on {c[c.capped].time.nunique()} bars; "
          f"share of the summed QLIKE of the proposed model in the forecast of {CRASH_BAR:%d %b %Y %H:%M}: "
          f"{c[c.time == CRASH_BAR].ql.sum() / c.ql.sum():.3f}; in the ten largest terms: {c.ql.nlargest(10).sum() / c.ql.sum():.3f}")
    rows = []
    for label, key in QLIKE_ROWS:
        r = dict(model=label)
        for name, test in out.items():
            per = test.groupby(["groupset", "model", "dataset"]).ql.mean().unstack("dataset")
            r[f"qlike_har ({name})"] = per.loc[key].mean() / per.loc[HAR].mean()
            r[f"coins_better ({name})"] = int((per.loc[key] < per.loc[HAR]).sum())
            if key != PROPOSED:
                loss = {kk: test[(test.groupset == kk[0]) & (test.model == kk[1])].pivot(index="time", columns="dataset", values="ql")
                        for kk in (PROPOSED, key)}
                t = loss[PROPOSED].dropna().index.intersection(loss[key].dropna().index)
                ratio = per.loc[PROPOSED].mean() / per.loc[key].mean()
                d = (loss[PROPOSED].loc[t] - loss[key].loc[t]).mean(axis=1)
                stat, p = diebold_mariano(d.to_numpy(), np.zeros(len(d)))
                r[f"prop_ratio ({name})"], r[f"prop_better ({name})"] = ratio, int((per.loc[PROPOSED] < per.loc[key]).sum())
                r[f"prop_dm ({name})"], r[f"prop_p ({name})"] = stat, p
        rows.append(r)
    t = pd.DataFrame(rows)
    for name in out:
        ok = t[f"prop_p ({name})"].notna()
        t.loc[ok, f"prop_p_holm ({name})"] = holm(t.loc[ok, f"prop_p ({name})"])
    t.to_csv(OUT / "v2_qlike_bar.csv", index=False)
    # how much of the summed loss the largest terms carry
    for label, key in [("proposed", PROPOSED), ("HAR", HAR), ("LightGBM-S", ("PMSE", "lgbms-pool"))]:
        g = out["all bars"]
        g = g[(g.groupset == key[0]) & (g.model == key[1])]
        total = g.ql.sum()
        bar = g[g.time == CRASH_BAR]
        top = g.nlargest(10, "ql")
        print(f"{label:12s} coin-bars {len(g)}, share of summed QLIKE in the forecast made at {CRASH_BAR:%d %b %Y %H:%M}: {bar.ql.sum() / total:.3f} "
              f"(ATOM alone {bar[bar.dataset == 'atom_4h'].ql.sum() / total:.3f}); ten largest terms {top.ql.sum() / total:.3f}, "
              f"of which in that bar {int((top.time == CRASH_BAR).sum())}")
    k = out["all bars"]
    k = k[(k.groupset == PROPOSED[0]) & (k.model == PROPOSED[1])].groupby(["dataset", "fold"]).k.first()
    print(f"level constant of the proposed model: median {k.median():.2f}, largest {k.max():.1f} ({k.idxmax()})")
    return t


# ============================================================================================ select
def select() -> pd.DataFrame:
    """Validation block of the first fold, which ends before the first test block: MSE relative to HAR of the
    models with and without the calendar term (seed-averaged forecasts, mean over coins)."""
    keys = [("HAR", HAR), ("LSTM", ("PMSE", "lstm-pool")), ("LSTM + calendar", ("PMSEC", "lstm+cal-pool")),
            ("Transformer", ("PMSE", "transformer-pool")), ("Transformer + calendar", ("PMSEC", "transformer+cal-pool")),
            ("H-LSTM, 2 layers", ("PMSE", "hl_meta-L2-pool")), ("H-LSTM, 2 layers + calendar", TWIN),
            ("H-QLSTM, 2 layers", ("PMSE", "hq_meta-L2-pool")), ("H-QLSTM, 2 layers + calendar", PROPOSED),
            ("H-QLSTM, 1 layer + calendar", ("PMSEC", "hq_meta+cal-pool")),
            ("H-LSTM, price and market, calendar", ("PMC", "hl_concat-L2+cal-pool")),
            ("LightGBM-S, pooled", ("PMSE", "lgbms-pool"))]
    ens = _splits([k for _, k in keys], QLIKE_RUNS)
    v = ens[(ens.split == "val") & (ens.fold == 0)].assign(se=lambda d: (d.pred - d.y) ** 2)
    print(f"validation block of the first fold: {v.time.min():%Y-%m-%d} to {v.time.max():%Y-%m-%d}")
    per = v.groupby(["groupset", "model", "dataset"]).se.mean().unstack("dataset")
    t = pd.DataFrame([dict(model=label, val_mse_har=per.loc[key].mean() / per.loc[HAR].mean(),
                           coins_better=int((per.loc[key] < per.loc[HAR]).sum())) for label, key in keys])
    t.to_csv(OUT / "v2_selection.csv", index=False)
    return t


# ============================================================================================ levels
def levels() -> pd.DataFrame:
    """Mean squared error of HAR and variance of the target on the test blocks, mean over coins."""
    har_runs = {"1h": "results/final_fb/out/final_single_1h_*", "4h": "results/final_fa/out/final_single_4h_*",
                "1d": "results/final_fa/out/final_single_1d_*"}
    rows = []
    for freq, pattern in har_runs.items():
        x = S.run(pattern)
        x = x[(x.model == "har") & (x.groupset == "P")].assign(se=lambda d: (d.pred - d.y) ** 2)
        g = x.groupby("dataset").agg(mse=("se", "mean"), var=("y", "var"), n=("y", "size"))
        rows.append(dict(bars=freq, mse_har=g.mse.mean(), var_target=g["var"].mean(), r2_har=(1 - g.mse / g["var"]).mean(),
                         bars_per_coin=int(g.n.median())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_levels.csv", index=False)
    return t


# ============================================================================================ exceed
EXCEED_ROWS = [("HAR", HAR), ("HAR-S", ("PMSE", "hars")), ("HAR-S, week cells", ("P", "hars_x")),
               ("LightGBM-S, pooled", ("PMSE", "lgbms-pool")), ("LightGBM-S, pooled, lagged exchange features", ("PM", "lgbmsl-pool")),
               ("H-LSTM, price and market, calendar", ("PMC", "hl_concat-L2+cal-pool")), ("H-LSTM, first design", ("PMC", "hl_gattn-pool")),
               ("H-LSTM + calendar (real twin)", TWIN), ("H-QLSTM + calendar (proposed)", PROPOSED)]


def exceed(rate: float = 0.05) -> pd.DataFrame:
    """A range limit from each forecast: the forecast log range plus a constant, the (1 - rate) quantile of the
    forecast errors on the validation block of the coin and fold. On the test blocks: how often the realised
    range exceeds the limit, the mean limit relative to that of HAR (a lower limit at the same exceedance
    rate ties up less margin), and the tick loss of the limit as a (1 - rate) quantile forecast of the log
    range (Koenker and Bassett 1978), relative to HAR, with the cross-coin DM test of the proposed model."""
    ens = _splits([k for _, k in EXCEED_ROWS], QLIKE_RUNS)
    by = ["dataset", "groupset", "model", "fold"]
    val = ens[ens.split == "val"].assign(e=lambda d: d.y - d.pred)
    c = val.groupby(by).e.quantile(1 - rate).rename("c")
    test = ens[ens.split == "test"].join(c, on=by)
    test["limit"] = test.pred + test.c
    test["hit"] = test.y > test.limit
    u = test.y - test.limit
    test["tick"] = u * ((1 - rate) - (u < 0))
    g = test.groupby(["groupset", "model", "dataset"]).agg(rate=("hit", "mean"), limit=("limit", lambda v: np.exp(v).mean()),
                                                            tick=("tick", "mean"))
    wide = {key: test[(test.groupset == key[0]) & (test.model == key[1])].pivot(index="time", columns="dataset", values="tick")
            for _, key in EXCEED_ROWS}
    rows = []
    for label, key in EXCEED_ROWS:
        a, h = g.loc[key], g.loc[HAR]
        # the limit is in units of the range, whose level differs between coins: mean of the per-coin ratios
        r = dict(model=label, exceedance=a.rate.mean(), limit_over_har=(a.limit / h.limit).mean(),
                 coins_lower_limit=int((a.limit < h.limit).sum()), tick_over_har=a.tick.mean() / h.tick.mean(),
                 coins_lower_tick=int((a.tick < h.tick).sum()))
        if key != PROPOSED:
            t = wide[PROPOSED].dropna().index.intersection(wide[key].dropna().index)
            d = (wide[PROPOSED].loc[t] - wide[key].loc[t]).mean(axis=1)
            r["prop_tick_ratio"] = g.loc[PROPOSED].tick.mean() / a.tick.mean()
            r["prop_dm"], r["prop_p"] = diebold_mariano(d.to_numpy(), np.zeros(len(d)))
        rows.append(r)
    t = pd.DataFrame(rows)
    ok = t.prop_p.notna()
    t.loc[ok, "prop_p_holm"] = holm(t.loc[ok, "prop_p"])
    t.to_csv(OUT / "v2_exceed.csv", index=False)
    return t


# ============================================================================================ lrselect
LR_RUNS = {   # learning rate -> runs that hold the pair at that rate (a later run overrides an earlier one)
    "one-layer": {3e-4: ["results/final_r7a/out/final_lr3e4_pool_4h_*"],
                  1e-3: ["results/final_fa/out/final_pool_4h_*", "results/v2/out/v2_depth_pool_4h_*"],
                  3e-3: ["results/final_r7a/out/final_lr3e3_pool_4h_*"]},
    "proposed": {1e-3: ["results/v2cal/out/v2_cal_pool_4h_*"], 3e-3: ["results/v2i/out/v2_lr3e3_cal_pool_4h_*"]},
}
LR_PAIRS = [("LSTM, 1 layer", "one-layer", ("PMSE", "qlstm-pool"), ("PMSE", "lstm-pool")),
            ("Hierarchical LSTM, 1 layer", "one-layer", ("PMSE", "hq_meta-pool"), ("PMSE", "hl_meta-pool")),
            ("Proposed pair", "proposed", PROPOSED, TWIN)]


def lrselect() -> pd.DataFrame:
    """Twin ratios with the learning rate chosen per model and fold on the validation block (the rate with the
    lowest validation MSE of the seed-averaged forecast over all coins), from the rates that were run."""
    rows = []
    for label, grid, a, b in LR_PAIRS:
        by_rate = {lr: _splits([a, b], runs) for lr, runs in LR_RUNS[grid].items()}
        parts, chosen = [], {}
        for key in (a, b):
            for fold in range(4):
                val = {}
                for lr, x in by_rate.items():
                    g = x[(x.groupset == key[0]) & (x.model == key[1]) & (x.fold == fold) & (x.split == "val")]
                    val[lr] = float(((g.pred - g.y) ** 2).mean())
                best = min(val, key=val.get)
                chosen.setdefault(key[1], []).append(best)
                x = by_rate[best]
                parts.append(x[(x.groupset == key[0]) & (x.model == key[1]) & (x.fold == fold) & (x.split == "test")])
        df = pd.concat(parts, ignore_index=True).assign(task="range", seed=0)
        v = Store([df]).pair(a, b)
        fixed = {lr: Store([x[x.split == "test"].assign(task="range", seed=0)]).pair(a, b)["ratio"] for lr, x in by_rate.items()}
        rows.append(dict(pair=label, ratio=v["ratio"], better=v["better"], p=v["p"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"],
                         equivalent=v["equivalent"], rates_quaternion=" ".join(f"{r:g}" for r in chosen[a[1]]),
                         rates_real=" ".join(f"{r:g}" for r in chosen[b[1]]),
                         **{f"ratio_at_{lr:g}": r for lr, r in fixed.items()}))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_lrselect.csv", index=False)
    return t


# ============================================================================================ rangein
def rangein(st: Store) -> pd.DataFrame:
    """Control of the first stage: the log range of the current bar in place of the zero real part of the price
    quaternion (group price_r), one-layer pooled models. Each model against itself on the standard groups, and
    the twin ratios with the range as an input."""
    (d,) = glob.glob("results/final_dr2/out/final_rangein_pool_4h_*")
    x = S.run(d)
    x = x[(x.groupset == "PRMSE") & (x.seed <= 2)]
    base = S.run("results/final_fa/out/final_pool_4h_*")
    base = base[(base.groupset == "PMSE") & (base.seed <= 2)]
    har = st.get(HAR)
    s2 = Store([x, base[base.model.isin(x.model.unique())], har])
    rows = []
    for m in sorted(x.model.unique()):
        v = s2.pair(("PRMSE", m), ("PMSE", m))
        rows.append(dict(contrast=f"{m}: range as input / zero real part", mse_har=s2.vs_har(("PRMSE", m)), ratio=v["ratio"],
                         better=v["better"], p=v["p"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"]))
    for q, r in [("qlstm", "lstm"), ("hq_meta", "hl_meta"), ("qtransformer", "transformer"), ("hq_gattn", "hl_gattn")]:
        for gs, name in (("PRMSE", "range as input"), ("PMSE", "zero real part")):
            v = s2.pair((gs, q + "-pool"), (gs, r + "-pool"))
            rows.append(dict(contrast=f"{q} / {r}, {name}", ratio=v["ratio"], better=v["better"], p=v["p"],
                             ci_lo=v["ci_lo"], ci_hi=v["ci_hi"]))
    t = pd.DataFrame(rows)
    ok = t.contrast.str.contains("range as input / zero")
    t.loc[ok, "p_holm"] = holm(t.loc[ok, "p"])
    t.to_csv(OUT / "v2_rangein.csv", index=False)
    return t


# ============================================================================================ tables
HEADER = "% generated by experiments/paper_v2_round2.py\n"
CHECK = r"\checkmark"
# ============================================================================================ depth
# (encoder, calendar term, depth label, real model, quaternion model, groupset)
DEPTH_ROWS = [
    ("LSTM / QLSTM", False, "1 layer", "lstm", "qlstm", "PMSE"), ("LSTM / QLSTM", False, "2 layers", "lstm-L2", "qlstm-L2", "PMSE"),
    ("LSTM / QLSTM", False, "3 layers", "lstm-L3", "qlstm-L3", "PMSE"),
    (r"H-LSTM / \HQ{}", False, "1 layer", "hl_meta", "hq_meta", "PMSE"), (r"H-LSTM / \HQ{}", False, "2 layers", "hl_meta-L2", "hq_meta-L2", "PMSE"),
    (r"H-LSTM / \HQ{}", True, "1 layer", "hl_meta+cal", "hq_meta+cal", "PMSEC"),
    (r"H-LSTM / \HQ{}", True, "2 layers", "hl_meta-L2+cal", "hq_meta-L2+cal", "PMSEC"),
    ("CNN / QCNN", False, "2 conv.", "cnn-L2", "qcnn-L2", "PMSE"), ("CNN / QCNN", False, "4 conv.", "cnn", "qcnn", "PMSE"),
    ("CNN / QCNN", False, "6 conv.", "cnn-L6", "qcnn-L6", "PMSE"),
    ("H-CNN / H-QCNN", False, "2 conv.", "hc_meta-L2", "hqc_meta-L2", "PMSE"), ("H-CNN / H-QCNN", False, "4 conv.", "hc_meta", "hqc_meta", "PMSE"),
]


def depth_tex(st: Store) -> pd.DataFrame:
    """MSE relative to HAR of every real and quaternion encoder of the depth grid by depth (20,000 parameters,
    pooled 4-hour bars), with the twin ratio and its Holm verdict from the family of all twin tests."""
    fam = pd.read_csv(OUT / "v2_twins_all.csv")
    rows = []
    for encoder, cal, depth, r, q, gs in DEPTH_ROWS:
        a, b = (gs, q + "-pool"), (gs, r + "-pool")
        pooled = S.pooled_ratio(st.get(a, b), a, b)[0]
        hit = fam[(fam.bars == "4h") & ((fam.pooled - pooled).abs() < 1e-9)]
        assert len(hit) == 1, (encoder, depth, len(hit))             # the same pair in the twin family
        rows.append(dict(encoder=encoder, calendar=cal, depth=depth, real=st.vs_har(b), quaternion=st.vs_har(a),
                         ratio=hit.ratio.iloc[0], better=int(hit.better.iloc[0]), p_holm=hit.p_holm.iloc[0]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_depth.csv", index=False)
    lines = [HEADER, r"\begin{tabular}{@{}llcrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{MSE / HAR} & & \\", r"\cmidrule(lr){4-5}",
             r"Encoders & Depth & Cal. & Real & Quat. & Q / R & Q better \\", r"\midrule"]
    last = None
    for _, r in t.iterrows():
        if last is not None and r.encoder != last:
            lines.append(r"\addlinespace[2pt]")
        name = r.encoder if r.encoder != last else ""
        last = r.encoder
        proposed = (r.encoder.startswith("H-LSTM") and r.calendar and r.depth == "2 layers")
        quat = rf"\textbf{{{r.quaternion:.3f}}}" if proposed else f"{r.quaternion:.3f}"
        star = "$^{*}$" if r.p_holm < 0.05 else ""
        lines.append(rf"{name} & {r.depth} & {CHECK if r.calendar else ''} & {r.real:.3f} & {quat} & {r.ratio:.3f}{star} & {r.better}/15 \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "v2_depth.tex").write_text("\n".join(lines) + "\n")
    return t


# (label, calendar, key); key None = group heading
MAIN_ROWS = [
    ("Benchmarks", None, None),
    ("Persistence", "", ("P", "persistence")),
    ("GARCH(1,1)", "", ("P", "garch")),
    ("GJR-GARCH", "", ("P", "gjr")),
    ("CARR(1,1)", "", ("P", "carr")),
    ("HAR", "", HAR),
    ("HAR-X", "", ("PMSE", "harx")),
    ("LightGBM, pooled", "", ("PMSE", "lgbm-pool")),
    ("HAR-S", CHECK, ("PMSE", "hars")),
    ("HAR-XS", CHECK, ("PMSE", "harxs")),
    (r"HAR-S$\times$", CHECK, ("P", "hars_x")),
    (r"HAR-XS$\times$", CHECK, ("PMSE", "harxs_x")),
    ("LightGBM-S, pooled", CHECK, ("PMSE", "lgbms-pool")),
    ("LightGBM-S, pooled, lagged", CHECK, ("PM", "lgbmsl-pool")),
    ("Flat networks", None, None),
    ("CNN, 6 convolutions", "", ("PMSE", "cnn-L6-pool")),
    ("LSTM", "", ("PMSE", "lstm-pool")),
    ("Transformer", "", ("PMSE", "transformer-pool")),
    ("LSTM", CHECK, ("PMSEC", "lstm+cal-pool")),
    ("Transformer", CHECK, ("PMSEC", "transformer+cal-pool")),
    ("Hierarchical networks, two layers", None, None),
    ("H-LSTM", "", ("PMSE", "hl_meta-L2-pool")),
    (r"\HQ{}", "", ("PMSE", "hq_meta-L2-pool")),
    ("H-LSTM (real twin)", CHECK, TWIN),
    (r"\HQ{} (proposed)", CHECK, PROPOSED),
    ("Real hierarchical networks on price and market data", None, None),
    ("H-LSTM, two layers", CHECK, ("PMC", "hl_concat-L2+cal-pool")),
    ("H-LSTM, first calendar design", CHECK, ("PMC", "hl_gattn-pool")),
]


def tick_har(ens: pd.DataFrame, rate: float = 0.05) -> pd.Series:
    """Tick loss of the (1 - rate) limit of every model relative to that of HAR, mean over coins (see exceed)."""
    by = ["dataset", "groupset", "model", "fold"]
    c = ens[ens.split == "val"].assign(e=lambda d: d.y - d.pred).groupby(by).e.quantile(1 - rate).rename("c")
    test = ens[ens.split == "test"].join(c, on=by)
    u = test.y - (test.pred + test.c)
    test["tick"] = u * ((1 - rate) - (u < 0))
    per = test.groupby(["groupset", "model", "dataset"]).tick.mean().unstack("dataset")
    return per.mean(axis=1) / per.loc[HAR].mean()


def main_tex(st: Store) -> pd.DataFrame:
    keys = [k for _, _, k in MAIN_ROWS if k is not None]
    ens = _splits(keys, QLIKE_RUNS)
    test = _calibrated(ens, None, caps=train_caps())        # QLIKE with the realised squared range capped
    per = test.groupby(["groupset", "model", "dataset"]).ql.mean().unstack("dataset")
    ql, tick = per.mean(axis=1) / per.loc[HAR].mean(), tick_har(ens)
    rows = []
    for label, cal, key in MAIN_ROWS:
        if key is None:
            rows.append(dict(label=label, heading=True))
            continue
        r = dict(label=label, calendar=cal, heading=False, key="/".join(key), mse=st.vs_har(key), qlike=ql.loc[key],
                 tick=tick.loc[key])
        if key != HAR:
            d = dm_table(st.get(key, HAR), {"range": HAR})
            r.update(better=int((d.rel_loss < 1).sum()), n=len(d))
        if key != PROPOSED:
            r.update({f"prop_{k}": v for k, v in st.pair(PROPOSED, key).items()})
        rows.append(r)
    t = pd.DataFrame(rows)
    ok = t.prop_p.notna()
    t.loc[ok, "prop_p_holm"] = holm(t.loc[ok, "prop_p"])
    t.to_csv(OUT / "v2_main.csv", index=False)
    lines = [HEADER, r"\begin{tabular}{@{}lcrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{4}{c}{Against HAR} & \multicolumn{3}{c}{Proposed / model (MSE)} \\",
             r"\cmidrule(lr){3-6}\cmidrule(l){7-9}",
             r"Model & Cal. & MSE & better & QLIKE & Limit & ratio & better & 90\% interval \\", r"\midrule"]
    first = True
    for _, r in t.iterrows():
        if r.heading:
            if not first:
                lines.append(r"\addlinespace[2pt]")
            lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{{r.label}}}}} \\")
            first = False
            continue
        better = "--" if pd.isna(r.get("better")) else f"{int(r.better)}/{int(r.n)}"
        if pd.isna(r.get("prop_ratio")):
            vs = "-- & -- & --"
        else:
            star = "$^{*}$" if r.prop_p_holm < 0.05 else ""
            vs = f"{r.prop_ratio:.3f}{star} & {int(r.prop_better)}/15 & {r.prop_ci_lo:.3f}--{r.prop_ci_hi:.3f}"
        bold = r"\textbf" if "proposed" in r.label else ""
        lines.append(rf"\quad {r.label} & {r.calendar} & {bold}{{{r.mse:.3f}}} & {better} & {r.qlike:.3f} & {r.tick:.3f} & {vs} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "v2_main.tex").write_text("\n".join(lines) + "\n")
    return t.drop(columns=["heading", "key"])


# calendar-aware models at every bar length: runs that hold them
ROBUST_RUNS = {
    "1h": ["results/final_fb/out/final_single_1h_*", "results/final_local/final_har_season_1h_*", "results/final_local/final_lgbm_1h_*",
           "results/v2local/v2_lgbm_seeds_1h_*", "results/r8_local/final_harsx_1h_r8", "results/r11_local/final_harxsx_1h_r11",
           "results/r8_local/final_lgbmlags_1h_r8", "results/v2f/out/v2_cal_pool_1h_*"],
    "4h": None,                  # the 4-hour store
    "1d": ["results/final_fa/out/final_single_1d_*", "results/final_local/final_har_season_1d_*", "results/final_local/final_lgbm_1d_*",
           "results/v2local/v2_lgbm_seeds_1d_*", "results/r8_local/final_lgbmlags_1d_r8", "results/v2f/out/v2_cal_pool_1d_*"],
}
ROBUST_ROWS = [("HAR-S", ("PMSE", "hars")), ("HAR-XS", ("PMSE", "harxs")), (r"HAR-S$\times$", ("P", "hars_x")),
               (r"HAR-XS$\times$", ("PMSE", "harxs_x")), ("LightGBM-S, pooled", ("PMSE", "lgbms-pool")),
               ("LightGBM-S, pooled, lagged", ("PM", "lgbmsl-pool")),
               ("LSTM", ("PMSEC", "lstm+cal-pool")), ("Transformer", ("PMSEC", "transformer+cal-pool")),
               ("H-LSTM (real twin)", TWIN), (r"\HQ{} (proposed)", PROPOSED)]


def robust_tex(st: Store) -> pd.DataFrame:
    """MSE relative to HAR of the calendar-aware models at every bar length and the proposed model against
    each of them (Holm adjustment within a bar length)."""
    rows = []
    for freq, runs in ROBUST_RUNS.items():
        if runs is None:
            s_ = st
        else:
            keep = {k for _, k in ROBUST_ROWS} | {HAR}
            parts = []
            for pattern in runs:
                x = S.run(pattern)
                parts.append(x[[(g, m) in keep for g, m in zip(x.groupset, x.model)]])
            s_ = Store([pd.concat(parts, ignore_index=True)])
        for label, key in ROBUST_ROWS:
            if not s_.has(key):
                rows.append(dict(freq=freq, label=label))
                continue
            r = dict(freq=freq, label=label, mse=s_.vs_har(key))
            if key != PROPOSED:
                r.update({f"prop_{k}": v for k, v in s_.pair(PROPOSED, key).items()})
            rows.append(r)
    t = pd.DataFrame(rows)
    t["prop_p_holm"] = float("nan")
    for freq, g in t.groupby("freq"):
        ok = g.prop_p.notna()
        t.loc[g.index[ok], "prop_p_holm"] = holm(g.prop_p[ok])
    t.to_csv(OUT / "v2_robust.csv", index=False)
    names = {"1h": "Hourly", "4h": "4-hourly", "1d": "Daily"}
    lines = [HEADER, r"\begin{tabular}{@{}l" + "rr" * 3 + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{names[f]}}}" for f in ROBUST_RUNS) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(3)),
             "Model & " + " & ".join(r"MSE / HAR & Prop. / model" for _ in ROBUST_RUNS) + r" \\", r"\midrule"]
    for label, _ in ROBUST_ROWS:
        cells = []
        for f in ROBUST_RUNS:
            r = t[(t.freq == f) & (t.label == label)].iloc[0]
            if pd.isna(r.get("mse")):
                cells.append("-- & --")
                continue
            vs = "--" if pd.isna(r.prop_ratio) else f"{r.prop_ratio:.3f}" + ("$^{*}$" if r.prop_p_holm < 0.05 else "")
            cells.append(f"{r.mse:.3f} & {vs}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "v2_robust.tex").write_text("\n".join(lines) + "\n")
    return t


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    if which in ("benchx_save", "lgbmlag16_save"):
        {"benchx_save": benchx_save, "lgbmlag16_save": lgbmlag16_save}[which]()
        sys.exit()
    store = Store(parts_4h()) if which in ("bench", "benchx", "twins", "width", "blocks", "small", "main", "robust", "scope", "depth", "mcs", "rangein", "all") else None
    for name, fn in [("main", lambda: main_tex(store)), ("robust", lambda: robust_tex(store)), ("bench", lambda: bench(store)), ("benchx", lambda: benchx(store)), ("mcs", lambda: mcs(store)), ("twins", lambda: twins(store)), ("depth", lambda: depth_tex(store)), ("width", lambda: width(store)),
                     ("blocks", lambda: blocks(store)), ("scope", lambda: scope(store)), ("small", lambda: small(store)), ("seeds", seeds),
                     ("returns", returns_pooled), ("qlike", qlike), ("select", select), ("levels", levels),
                     ("exceed", exceed), ("lrselect", lrselect), ("rangein", lambda: rangein(store))]:
        if which in (name, "all"):
            print(f"\n=== {name}")
            print(fn().round(4).to_string(index=False))
