"""
Checks asked for by review round 1 of paper v2 (docs/paper/review/v2_round1/), computed from saved
forecasts only (pooled 4-hour bars, 15 coins, seeds 0-2 unless stated).

    .venv/bin/python -W ignore experiments/paper_v2_checks.py [seeds|qlike|sources|returns|all]

  seeds    tables/v2_seed_sensitivity.csv  twin ratios of single seeds (same-seed and all nine pairings)
                                            and equivalence at margins of 0.5%, 1% and 2% (REV-04, REV-15);
           tables/v2_single_seed.csv        MSE/HAR of single-seed forecasts against the three-seed
                                            average, networks and LightGBM-S (REV-16)
  qlike    tables/v2_qlike.csv              calibrated QLIKE relative to HAR with one level constant per
                                            coin and fold (the paper's definition) and with one constant
                                            per coin, fold and time of day; cross-coin DM test of the
                                            proposed model against each model on both (REV-05)
  sources  tables/v2_sources_v1.csv         16 against 8 features in the earlier pooled grid (REV-17)
  returns  tables/v2_returns.csv            next-bar log return, the negative control (REV-09)
"""

import glob
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from src.paper.evaluate import _to_time, diebold_mariano, dm_cross, dm_table, holm, load_run  # noqa: E402
import paper_v2_summary as S  # noqa: E402
import paper_v2_tables as T  # noqa: E402
from paper_round8_checks import _ratio_se  # noqa: E402

OUT = T.OUT
HAR, PROPOSED = S.HAR, T.PROPOSED
TWIN = ("PMSEC", "hl_meta-L2+cal-pool")
LGBMS = ("PMSE", "lgbms-pool")


def _pooled_loss(df: pd.DataFrame, key: tuple, seed: int | None = None) -> pd.Series:
    """Coin-averaged squared error per date of one model: one seed, or the average forecast of the seeds."""
    g = df[(df.groupset == key[0]) & (df.model == key[1])]
    if seed is not None:
        g = g[g.seed == seed]
    g = g.groupby(["dataset", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
    g["se"] = (g.pred - g.y) ** 2
    return g.pivot(index="time", columns="dataset", values="se").dropna().mean(axis=1)


def _ratio(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    t = a.index.intersection(b.index)
    r, _, se = _ratio_se(a.loc[t], b.loc[t])
    return r, r - 1.645 * se, r + 1.645 * se


def seeds():
    df = T.load()
    rows = []
    pairs = [(f"{fam}, {depth}", (gs, q + "-pool"), (gs, r + "-pool")) for fam, depth, q, r, gs in S.TWIN_ROWS]
    for label, a, b in pairs:
        have = set(zip(df.groupset, df.model))
        if a not in have or b not in have:
            continue
        r, lo, hi = _ratio(_pooled_loss(df, a), _pooled_loss(df, b))
        same = [_ratio(_pooled_loss(df, a, s), _pooled_loss(df, b, s)) for s in (0, 1, 2)]
        cross = [_ratio(_pooled_loss(df, a, s), _pooled_loss(df, b, u)) for s, u in product((0, 1, 2), repeat=2)]
        inside = lambda lo_, hi_, m: lo_ > 1 - m and hi_ < 1 + m
        rows.append(dict(pair=label, ratio=r, ci_lo=lo, ci_hi=hi,
                         equiv_0p5=inside(lo, hi, 0.005), equiv_1=inside(lo, hi, 0.01), equiv_2=inside(lo, hi, 0.02),
                         same_seed_min=min(x[0] for x in same), same_seed_max=max(x[0] for x in same),
                         same_seed_equiv_1=sum(inside(x[1], x[2], 0.01) for x in same),
                         pairing_min=min(x[0] for x in cross), pairing_max=max(x[0] for x in cross),
                         pairings_equiv_1=sum(inside(x[1], x[2], 0.01) for x in cross)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_seed_sensitivity.csv", index=False)
    print(t.round(4).to_string(index=False))

    # single-seed forecasts against the three-seed average, relative to HAR (mean over coins of the ratio)
    har = df[(df.groupset == HAR[0]) & (df.model == HAR[1])]
    rows = []
    for label, key in [("H-QLSTM (proposed)", PROPOSED), ("H-LSTM (real twin)", TWIN), ("LightGBM-S, pooled", LGBMS),
                       ("LSTM + calendar", ("PMSEC", "lstm+cal-pool")), ("Transformer + calendar", ("PMSEC", "transformer+cal-pool"))]:
        g = df[(df.groupset == key[0]) & (df.model == key[1])]
        r = dict(model=label)
        for s in (0, 1, 2):
            d = dm_table(pd.concat([har, g[g.seed == s]]), {"range": HAR})
            r[f"seed{s}"] = d.rel_loss.mean()
        r["average_of_seeds"] = dm_table(pd.concat([har, g]), {"range": HAR}).rel_loss.mean()
        rows.append(r)
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / "v2_single_seed.csv", index=False)
    print()
    print(t2.round(4).to_string(index=False))
    # single proposed network against single LightGBM-S fits, and ensembles against ensembles
    for s in (0, 1, 2):
        both = np.array([(g, m) in (PROPOSED, LGBMS) for g, m in zip(df.groupset, df.model)])
        x = df[both & (df.seed == s).to_numpy()]
        c = dm_cross(x, PROPOSED, LGBMS)
        print(f"seed {s}: proposed / LightGBM-S = {dm_table(x, {'range': LGBMS}).rel_loss.mean():.4f}, DM {c['dm_stat']:+.2f}, p = {c['p']:.3f}")


QLIKE_MODELS = [("HAR", HAR), ("HAR-S", ("PMSE", "hars")), ("HAR-XS", ("PMSE", "harxs")), ("LightGBM-S, pooled", LGBMS),
                ("LSTM", ("PMSE", "lstm-pool")), ("Transformer", ("PMSE", "transformer-pool")),
                ("H-LSTM, 2 layers", ("PMSE", "hl_meta-L2-pool")), ("H-QLSTM, 2 layers", ("PMSE", "hq_meta-L2-pool")),
                ("LSTM + calendar", ("PMSEC", "lstm+cal-pool")), ("Transformer + calendar", ("PMSEC", "transformer+cal-pool")),
                ("H-LSTM + calendar (real twin)", TWIN), ("H-QLSTM + calendar (proposed)", PROPOSED)]


def _splits(keys: set) -> pd.DataFrame:
    """Seed-averaged validation and test forecasts of the models in `keys` (as calibrated_qlike in paper_v2_tables)."""
    files = {}
    for pattern in T.MAIN_RUNS:
        (directory,) = glob.glob(pattern)
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
    return (df.groupby(["dataset", "groupset", "model", "fold", "split", "time"], as_index=False)
              .agg(y=("y", "first"), pred=("pred", "mean")))


def qlike():
    """QLIKE on the variance scale needs a level: the networks forecast the log range, so exp(2 * forecast) is
    not a mean variance. The paper rescales it with one constant per coin and fold from the validation block.
    If the forecast error variance depends on the time of day, one constant leaves a calendar-dependent bias
    that only the models with calendar information expose; a constant per time-of-day cell removes it."""
    ens = _splits({k for _, k in QLIKE_MODELS})
    ens["cell"] = ens.time.dt.hour                       # time of day of the target bar's open (six cells on 4h bars)
    ens["s2"], ens["raw"] = np.exp(2 * ens.y), np.exp(2 * ens.pred)
    out = {}
    for name, by in [("one constant", ["dataset", "groupset", "model", "fold"]),
                     ("per time of day", ["dataset", "groupset", "model", "fold", "cell"])]:
        val = ens[ens.split == "val"].groupby(by).agg(s2=("s2", "mean"), raw=("raw", "mean"))
        k = (val.s2 / val.raw).rename("k")
        test = ens[ens.split == "test"].join(k, on=by)
        test["h"] = test.k * test.raw
        test["ql"] = test.s2 / test.h - np.log(test.s2 / test.h) - 1
        out[name] = test
    rows = []
    for label, key in QLIKE_MODELS:
        r = dict(model=label)
        for name, test in out.items():
            per = test.groupby(["groupset", "model", "dataset"]).ql.mean().unstack("dataset")
            r[f"qlike_har ({name})"] = (per.loc[key] / per.loc[HAR]).mean()
            if key != PROPOSED:
                loss = {kk: test[(test.groupset == kk[0]) & (test.model == kk[1])].pivot(index="time", columns="dataset", values="ql")
                        for kk in (PROPOSED, key)}
                t = loss[PROPOSED].dropna().index.intersection(loss[key].dropna().index)
                d = (loss[PROPOSED].loc[t] - loss[key].loc[t]).mean(axis=1)
                stat, p = diebold_mariano(d.to_numpy(), np.zeros(len(d)))
                r[f"proposed_vs_model_dm ({name})"], r[f"p ({name})"] = stat, p
        rows.append(r)
    t = pd.DataFrame(rows)
    for name in out:
        ok = t[f"p ({name})"].notna()
        t.loc[ok, f"p_holm ({name})"] = holm(t.loc[ok, f"p ({name})"])
    t.to_csv(OUT / "v2_qlike.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 30)
    print(t.round(4).to_string(index=False))
    # the spread of the level constant across the six time-of-day cells: how calendar-dependent the level is
    test = out["per time of day"]
    spread = test.groupby(["groupset", "model", "dataset", "fold"]).k.agg(lambda v: v.max() / v.min())
    print("\nmax/min of the level constant over the six time-of-day cells (median over coins and folds):")
    for label, key in QLIKE_MODELS:
        print(f"  {label:32s} {spread.loc[key].median():.2f}")


def sources():
    """16 features (price, market, social, sentiment) against 8 (price, market) in the earlier pooled 4h grid."""
    (d,) = glob.glob("results/final_fa/out/final_pool_4h_*")
    df = load_run(d)
    df = df[(df.task == "range") & (df.seed <= 2)]
    rows = []
    for model in sorted(df.model.unique()):
        a, b = ("PMSE", model), ("PM", model)
        x = df[(df.model == model) & df.groupset.isin(["PM", "PMSE"])]
        if x.groupset.nunique() < 2:
            continue
        t = dm_table(x, {"range": b})
        c = dm_cross(x, a, b)
        rows.append(dict(model=model, ratio_16_over_8=t.rel_loss.mean(), coins_16_better=int((t.rel_loss < 1).sum()), n=len(t),
                         dm=c["dm_stat"], p=c["p"]))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    t.to_csv(OUT / "v2_sources_v1.csv", index=False)
    print(t.round(4).to_string(index=False))


def returns():
    """Next-bar log return, single-coin 4h models of the earlier grid: MSE relative to a zero forecast."""
    (d,) = glob.glob("results/final_fa/out/final_return_4h_*")
    df = load_run(d)
    df = df[(df.task == "return") & (df.seed <= 2)]
    ens = df.groupby(["dataset", "groupset", "model", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
    ens["se"], ens["se0"] = (ens.pred - ens.y) ** 2, ens.y ** 2
    g = ens.groupby(["groupset", "model", "dataset"]).agg(mse=("se", "mean"), mse0=("se0", "mean"))
    g["rel"] = g.mse / g.mse0
    t = g.groupby(["groupset", "model"]).agg(mse_over_zero=("rel", "mean"), coins_better=("rel", lambda v: int((v < 1).sum())),
                                              n=("rel", "size")).reset_index()
    t.to_csv(OUT / "v2_returns.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("seeds", seeds), ("qlike", qlike), ("sources", sources), ("returns", returns)]:
        if which in (name, "all"):
            print(f"\n=== {name}")
            fn()
