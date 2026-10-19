"""
Checks for the sixth fresh simulated review (pooled and single-coin range forecasts).

    .venv/bin/python -W ignore experiments/paper_fresh6_checks.py

1. Pooled LightGBM-S against the best network without calendar inputs of each frequency
   (MSE, five-seed network ensembles as in Tables 3 and S1-S3): mean of per-coin ratios,
   coins, cross-coin DM.
2. Equivalence-margin sensitivity: twin pairs whose 90% interval lies inside +-0.5%, +-1%
   and +-2% (from review_equivalence.csv).
3. Calibrated QLIKE comparisons quoted in the text, with a Wilcoxon signed-rank test over the
   15 per-coin log loss ratios, which is not dominated by the few extreme bars that make the
   DM statistic unreliable for QLIKE.
4. Provenance of three numbers that had no shipped table: the composition of the model
   confidence set universe, the LunarCrush price-field statistics (current snapshot and the
   February 2026 pull in data/cache/), and the bars with zero range.
Outputs: docs/paper/manuscript/tables/f6_lgbm_pairs.csv, f6_equiv_margins.csv, f6_qlike_wilcoxon.csv,
         f6_mcs_universe.csv, f6_lunarcrush_prices.csv, f6_zero_range.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from paper_tables_final import OUT_T, _pair, run, setting  # noqa: E402

BEST = {"4h": ("PM", "hl_gattn-pool"), "1h": ("PM", "hl_gattn-pool"), "1d": ("PMSE", "ht_concat-pool")}


def lgbm_pairs():
    rows = []
    for f, best in BEST.items():
        df = setting(f)
        r = _pair(df, ("PM", "lgbms-pool"), best)
        rows.append(dict(freq=f, a="lgbms-pool PM", b=f"{best[1]} {best[0]}", **r))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "f6_lgbm_pairs.csv", index=False)
    print(t.round(4).to_string(index=False))


def equiv_margins():
    e = pd.read_csv(OUT_T / "review_equivalence.csv")
    e = e[e.setting.isin(["single", "pooled"])]
    rows = []
    for (f, s), g in e.groupby(["freq", "setting"], sort=False):
        rec = dict(freq=f, setting=s, pairs=len(g))
        for m in (0.005, 0.01, 0.02):
            rec[f"equiv_{m:g}"] = int(((g.ci_lo > 1 - m) & (g.ci_hi < 1 + m)).sum())
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "f6_equiv_margins.csv", index=False)
    print(t.to_string(index=False))


QL_PAIRS = {  # (a, b): calibrated QLIKE on the squared range, seeds 0-2
    "LightGBM-S / H-LSTM (gattn) PMC": (("PM", "lgbms-pool"), ("PMC", "hl_gattn-pool")),
    "LightGBM-S / H-Transformer (concat) PMC": (("PM", "lgbms-pool"), ("PMC", "ht_concat-pool")),
    "Transformer pooled PMSE / HAR": (("PMSE", "transformer-pool"), ("P", "har")),
    "H-LSTM (gattn) PMC / HAR": (("PMC", "hl_gattn-pool"), ("P", "har")),
    "HAR-S / HAR": (("PMSE", "hars"), ("P", "har")),
    "Transformer pooled / H-Transformer (meta) pooled": (("PMSE", "transformer-pool"), ("PMSE", "ht_meta-pool")),
}


def qlike_wilcoxon():
    from paper_review_checks import RUNS, calibrated_variance, ensemble, load_both, qlike_loss
    keep = {k for ab in QL_PAIRS.values() for k in ab}
    rows = []
    for f in ("4h", "1h", "1d"):
        E = ensemble(load_both(RUNS[f] + [f"final_calendar_pool_{f}", f"final_calendar2_pool_{f}"]), seeds=[0, 1, 2])
        E = E[[(g, m) in keep for g, m in zip(E.groupset, E.model)]]
        V = calibrated_variance(E)
        V["ql"] = qlike_loss(V.s2, V.h)
        per = V.groupby(["groupset", "model", "dataset"]).ql.mean()
        for name, (a, b) in QL_PAIRS.items():
            if a not in per.droplevel("dataset").index or b not in per.droplevel("dataset").index:
                continue
            r = (per.loc[a] / per.loc[b]).dropna()
            p = wilcoxon(np.log(r)).pvalue
            rows.append(dict(freq=f, pair=name, ratio=r.mean(), better=int((r < 1).sum()), n=len(r),
                             median=r.median(), p_wilcoxon=p))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "f6_qlike_wilcoxon.csv", index=False)
    print(t.round(4).to_string(index=False))


def mcs_universe():
    """Composition of the MCS universe of review_checks.main_select (same filters, no bootstrap)."""
    from paper_review_checks import RUNS, ensemble, load_both
    rows = []
    for f in ("4h", "1h", "1d"):
        E = ensemble(load_both(RUNS[f]))
        E = E[~E.model.isin(["mean"]) & (E.split == "test")]
        E["se"] = (E.pred - E.y) ** 2
        L = E.pivot_table(index="time", columns=["groupset", "model"], values="se", aggfunc="mean").dropna(axis=1)
        drop = {"persistence", "ridge", "garch", "gjr"}
        L = L.loc[:, [c for c in L.columns if c[1] not in drop and not (c[1] == "har" and c[0] != "P")]]
        L = L.T.drop_duplicates().T
        kind = ["gradient boosting" if m.startswith("lgbm") else "HAR family" if m.startswith("har") else "network"
                for _, m in L.columns]
        c = pd.Series(kind).value_counts()
        rows.append(dict(freq=f, models=L.shape[1], **{k: int(c.get(k, 0)) for k in
                                                        ("network", "HAR family", "gradient boosting")}))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "f6_mcs_universe.csv", index=False)
    print(t.to_string(index=False))


def lunarcrush_prices():
    """LunarCrush's own daily OHLC (not used in the paper): close == high, high below max(open, close),
    and the correlation of low/close - 1 with LunarCrush's next-day close-to-close return, which
    is mechanical when the daily bar covers the following 24 hours; hourly bars with O = H = L = C."""
    rows = []
    for snap, fmt in (("2026-10 (used)", "data/lunarcrush/{c}_{f}.csv"),
                      ("2026-02 (earlier pull)", "data/cache/lunarcrush_{c}_{f}_full.csv")):
        for c in ("btc", "eth", "sol"):
            d = pd.read_csv(fmt.format(c=c, f="day"), index_col=0, parse_dates=True)
            d.index = pd.to_datetime(d.index, utc=True)
            x, r = d.low / d.close - 1, np.log(d.close).diff().shift(-1)
            early, late = d.index.year <= 2022, d.index.year >= 2023
            h = pd.read_csv(fmt.format(c=c, f="hour"), index_col=0, parse_dates=True)
            h.index = pd.to_datetime(h.index, utc=True)
            flat = ((h.open == h.high) & (h.high == h.low) & (h.low == h.close))[h.index.year <= 2021]
            rows.append(dict(snapshot=snap, coin=c, days=len(d), close_eq_high=np.isclose(d.close, d.high).mean(),
                             high_below_open_close=(d.high < np.maximum(d.open, d.close) - 1e-9).mean(),
                             rho_2020_22=x[early].corr(r[early]), rho_2023_on=x[late].corr(r[late]),
                             hourly_flat_2020_21=flat.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "f6_lunarcrush_prices.csv", index=False)
    print(t.round(4).to_string(index=False))


def zero_range():
    """Bars with zero range (target floored), per frequency and split (paper_round4_checks.zero_range)."""
    from paper_round4_checks import COINS, START
    from src.paper.data import make_dataset, walk_forward_folds
    floor = np.log(1e-6)
    rows = []
    for f in ("4h", "1h", "1d"):
        for coin in COINS:
            d = make_dataset(coin, f, start=START.get(f))
            fo = walk_forward_folds(d.frame.index, 4, 135, 180)
            zero = np.isclose(d.frame["y_range"].to_numpy(), floor)
            test = np.concatenate([k["test"] for k in fo])
            val = np.concatenate([k["val"] for k in fo])
            for split, idx in (("train (fold 4)", fo[-1]["train"]), ("val (all folds)", val), ("test", test)):
                rows.append(dict(freq=f, coin=coin, split=split, bars=len(idx), zero=int(zero[idx].sum())))
    t = pd.DataFrame(rows).groupby(["freq", "split"], as_index=False)[["bars", "zero"]].sum()
    t.to_csv(OUT_T / "f6_zero_range.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pd.set_option("display.width", 200)
        for name in sys.argv[1:]:
            globals()[name]()
        sys.exit()
    pd.set_option("display.width", 200)
    lgbm_pairs()
    equiv_margins()
    qlike_wilcoxon()
