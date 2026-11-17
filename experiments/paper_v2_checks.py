"""
Checks asked for by review round 1 of paper v2 (docs/paper/review/v2_round1/), computed from saved
forecasts only (pooled 4-hour bars, 15 coins, seeds 0-2 unless stated).

    .venv/bin/python -W ignore experiments/paper_v2_checks.py [seeds|sources|returns|vintages|all]

  seeds    tables/v2_seed_sensitivity.csv  twin ratios of single seeds (same-seed and all nine pairings)
                                            and equivalence at margins of 0.5%, 1% and 2% (REV-04, REV-15);
           tables/v2_single_seed.csv        MSE/HAR of single-seed forecasts against the three-seed
                                            average, networks and LightGBM-S (REV-16)
  sources  tables/v2_sources_v1.csv         16 against 8 features in the earlier pooled grid (REV-17)
  returns  tables/v2_returns.csv            next-bar log return, the negative control (REV-09)
  vintages tables/v2_vintages.csv           two downloads of the vendor history compared field by field
"""

import glob
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from src.paper.evaluate import dm_cross, dm_table, holm, load_run  # noqa: E402
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

    # single-seed forecasts against the three-seed average, relative to HAR (MSE over all coins and test bars)
    har = df[(df.groupset == HAR[0]) & (df.model == HAR[1])]
    rows = []
    for label, key in [("H-QLSTM (proposed)", PROPOSED), ("H-LSTM (real twin)", TWIN), ("LightGBM-S, pooled", LGBMS),
                       ("LSTM + calendar", ("PMSEC", "lstm+cal-pool")), ("Transformer + calendar", ("PMSEC", "transformer+cal-pool"))]:
        g = df[(df.groupset == key[0]) & (df.model == key[1])]
        r = dict(model=label)
        for s in (0, 1, 2):
            r[f"seed{s}"] = S.pooled_ratio(pd.concat([har, g[g.seed == s]]), key, HAR)[0]
        r["average_of_seeds"] = S.pooled_ratio(pd.concat([har, g]), key, HAR)[0]
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
        print(f"seed {s}: proposed / LightGBM-S = {S.pooled_ratio(x, PROPOSED, LGBMS)[0]:.4f}, DM {c['dm_stat']:+.2f}, p = {c['p']:.3f}")


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
        rows.append(dict(model=model, ratio_16_over_8=S.pooled_ratio(x, a, b)[0], coins_16_better=int((t.rel_loss < 1).sum()), n=len(t),
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
    t = g.groupby(["groupset", "model"]).agg(mse=("mse", "mean"), mse0=("mse0", "mean"), coins_better=("rel", lambda v: int((v < 1).sum())),
                                              n=("rel", "size")).reset_index()
    t.insert(2, "mse_over_zero", t.pop("mse") / t.pop("mse0"))       # losses averaged over the coins, then the ratio
    t.to_csv(OUT / "v2_returns.csv", index=False)
    print(t.round(4).to_string(index=False))


VENDOR_FIELDS = ["contributors_active", "contributors_created", "posts_active", "posts_created", "sentiment", "galaxy_score",
                 "social_dominance", "interactions"]


def vintages(first_test_bar: str = "2025-04-22"):
    """Two downloads of the vendor history compared: hourly values of February 2026 (data/cache, five coins) against
    those of the snapshot of the paper (data/lunarcrush, October 2026), from the first test bar to the end of the
    earlier download. `ratio_of_means` is the later download over the earlier one."""
    rows = []
    for old in sorted(glob.glob("data/cache/lunarcrush_*_hour_full.csv")):
        coin = os.path.basename(old).split("_")[1]
        new = f"data/lunarcrush/{coin}_hour.csv"
        if not os.path.exists(new):
            continue
        a = pd.read_csv(old, parse_dates=["Datetime"]).set_index("Datetime")
        b = pd.read_csv(new, parse_dates=["Datetime"]).set_index("Datetime")
        idx = a.index.intersection(b.index)
        idx = idx[(idx >= pd.Timestamp(first_test_bar, tz="UTC")) & (idx <= a.index.max())]
        for f in VENDOR_FIELDS:
            x, y = a.loc[idx, f].astype(float), b.loc[idx, f].astype(float)
            ok = x.notna() & y.notna()
            x, y = x[ok], y[ok]
            rel = (y - x).abs() / x.abs().clip(lower=1e-9)
            rows.append(dict(coin=coin, field=f, bars=len(x), last=str(a.index.max())[:13], corr=np.corrcoef(x, y)[0, 1],
                             identical=float((x == y).mean()), median_rel_diff=float(rel.median()),
                             p90_rel_diff=float(rel.quantile(0.9)), ratio_of_means=float(y.mean() / x.mean())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_vintages.csv", index=False)
    print(t.groupby("field").agg(corr_min=("corr", "min"), corr_max=("corr", "max"), identical_min=("identical", "min"),
                                 identical_max=("identical", "max"), median_rel_min=("median_rel_diff", "min"),
                                 median_rel_max=("median_rel_diff", "max"), ratio_min=("ratio_of_means", "min"),
                                 ratio_max=("ratio_of_means", "max")).round(4).to_string())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("seeds", seeds), ("sources", sources), ("returns", returns), ("vintages", vintages)]:
        if which in (name, "all"):
            print(f"\n=== {name}")
            fn()
