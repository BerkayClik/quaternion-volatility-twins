"""
Robustness checks requested by the (simulated) review, computed from saved forecasts only.

    .venv/bin/python experiments/paper_review_checks.py qlike     # calibrated QLIKE / variance-scale MSE
    .venv/bin/python experiments/paper_review_checks.py select    # validation-selected models, MCS
    .venv/bin/python experiments/paper_review_checks.py seeds     # seed spread of twin ratios, seed-matched pooling
    .venv/bin/python experiments/paper_review_checks.py norm_tests  # round 5: HAR-normalised cross-coin DM / TOST
    .venv/bin/python experiments/paper_review_checks.py insanity    # round 5: clipped HAR-X / HAR-XS under Table 9 losses
    .venv/bin/python experiments/paper_review_checks.py lgbm_seeds  # round 5: 3-seed pooled LightGBM-S

Outputs go to docs/paper/manuscript/tables/review_*.csv (and .tex where used in the paper).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from paper_tables_final import OUT_T, POOL, run_dir  # noqa: E402
from src.paper.evaluate import _to_time, diebold_mariano  # noqa: E402

RUNS = {f: [f"final_single_{f}", f"final_pool_{f}", f"final_harx_{f}", f"final_har_season_{f}", f"final_lgbm_{f}"]
        for f in ("4h", "1h", "1d")}


def load_both(names):
    """Long frame with test and validation forecasts of every cell (range task)."""
    parts = []
    for name in names:
        d = run_dir(name)
        if d is None:
            continue
        for path in glob.glob(os.path.join(d, "preds", "*.npz")):
            ds, task, gs, model, seed, fold = os.path.basename(path)[:-4].split("__")
            if task != "range":
                continue
            z = np.load(path)
            for split in ("test", "val"):
                parts.append(pd.DataFrame({
                    "dataset": ds, "groupset": gs, "model": model, "seed": int(seed[1:]), "fold": int(fold[1:]),
                    "split": split, "time": _to_time(z[f"{split}_time"]), "y": z[f"{split}_y"].astype(float),
                    "pred": z[f"{split}_pred"].astype(float)}))
    return pd.concat(parts, ignore_index=True)


def ensemble(df, seeds=None):
    if seeds is not None:
        df = df[df.seed.isin(seeds)]
    return (df.groupby(["dataset", "groupset", "model", "fold", "split", "time"], as_index=False)
              .agg(y=("y", "first"), pred=("pred", "mean")))


def calibrated_variance(ens):
    """Variance forecasts exp(2*pred) rescaled per (dataset, model, fold) so that their mean
    matches the mean squared range exp(2*y) on that fold's validation block."""
    out = []
    for key, g in ens.groupby(["dataset", "groupset", "model", "fold"]):
        v, t = g[g.split == "val"], g[g.split == "test"]
        c = np.mean(np.exp(2 * v.y)) / np.mean(np.exp(2 * v.pred))
        out.append(t.assign(h=c * np.exp(2 * t.pred), s2=np.exp(2 * t.y)))
    return pd.concat(out, ignore_index=True)


def qlike_loss(s2, h):
    return s2 / h - np.log(s2 / h) - 1


def qlike(freq):
    """Calibrated QLIKE and variance-scale MSE relative to HAR (P, har), and the key
    architecture / algebra comparisons under calibrated QLIKE."""
    V = calibrated_variance(ensemble(load_both(RUNS[freq])))
    V["ql"] = qlike_loss(V.s2, V.h)
    V["msev"] = (V.s2 - V.h) ** 2
    key = ["groupset", "model"]
    base = V[(V.groupset == "P") & (V.model == "har")].set_index(["dataset", "time"])
    rows = []
    for (gs, m), g in V.groupby(key):
        g = g.set_index(["dataset", "time"])
        idx = g.index.intersection(base.index)
        per = []
        for ds in g.index.get_level_values(0).unique():
            a, b = g.loc[ds], base.loc[ds]
            t = a.index.intersection(b.index)
            per.append((a.loc[t, "ql"].mean() / b.loc[t, "ql"].mean(), a.loc[t, "msev"].mean() / b.loc[t, "msev"].mean()))
        per = np.array(per)
        rows.append(dict(groupset=gs, model=m, qlike_cal=per[:, 0].mean(), qlike_better=int((per[:, 0] < 1).sum()),
                         msev=per[:, 1].mean(), n=len(per)))
    t = pd.DataFrame(rows).sort_values("qlike_cal")
    t.to_csv(OUT_T / f"review_qlike_{freq}.csv", index=False)
    return t, V


def pair_ratio(V, a, b, col="ql"):
    """Coin-mean ratio of the mean loss of a over b, coins a better, cross-coin DM."""
    A = V[(V.groupset == a[0]) & (V.model == a[1])].set_index(["dataset", "time"])[col]
    B = V[(V.groupset == b[0]) & (V.model == b[1])].set_index(["dataset", "time"])[col]
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    r = (A.groupby(level=0).mean() / B.groupby(level=0).mean())
    d = (A - B).groupby(level=1).mean().sort_index()
    stat, p = diebold_mariano(d.to_numpy(), np.zeros(len(d)))
    return dict(ratio=r.mean(), better=int((r < 1).sum()), n=len(r), dm=stat, p=p)


PAIRS = {  # architecture and algebra comparisons, PMSE inputs
    "H-LSTM / LSTM": (("PMSE", "hl_meta"), ("PMSE", "lstm")),
    "H-Transformer / Transformer": (("PMSE", "ht_meta"), ("PMSE", "transformer")),
    "pooled H-LSTM / LSTM": (("PMSE", "hl_meta-pool"), ("PMSE", "lstm-pool")),
    "pooled H-Transformer / Transformer": (("PMSE", "ht_meta-pool"), ("PMSE", "transformer-pool")),
    "QLSTM / LSTM": (("PMSE", "qlstm"), ("PMSE", "lstm")),
    "QTransformer / Transformer": (("PMSE", "qtransformer"), ("PMSE", "transformer")),
    "H-QLSTM / H-LSTM": (("PMSE", "hq_meta"), ("PMSE", "hl_meta")),
    "H-QTransformer / H-Transformer": (("PMSE", "hqt_meta"), ("PMSE", "ht_meta")),
    "pooled QLSTM / LSTM": (("PMSE", "qlstm-pool"), ("PMSE", "lstm-pool")),
    "pooled QTransformer / Transformer": (("PMSE", "qtransformer-pool"), ("PMSE", "transformer-pool")),
    "pooled H-QLSTM / H-LSTM": (("PMSE", "hq_meta-pool"), ("PMSE", "hl_meta-pool")),
    "pooled H-QTransformer / H-Transformer": (("PMSE", "hqt_meta-pool"), ("PMSE", "ht_meta-pool")),
}


def main_qlike():
    rows = []
    for f in ("4h", "1h", "1d"):
        t, V = qlike(f)
        print(f"== {f} calibrated QLIKE / HAR\n{t.head(25).round(3).to_string(index=False)}")
        for name, (a, b) in PAIRS.items():
            if ((V.groupset == a[0]) & (V.model == a[1])).any() and ((V.groupset == b[0]) & (V.model == b[1])).any():
                rows.append(dict(freq=f, pair=name, **pair_ratio(V, a, b)))
    p = pd.DataFrame(rows)
    p.to_csv(OUT_T / "review_qlike_pairs.csv", index=False)
    print(p.round(3).to_string(index=False))



def main_select():
    """Validation-based model selection per frequency and a Model Confidence Set
    (Hansen, Lunde and Nason 2011) on the coin-averaged squared-error losses."""
    from arch.bootstrap import MCS
    rows, mcs_rows = [], []
    for f in ("4h", "1h", "1d"):
        E = ensemble(load_both(RUNS[f]))
        E = E[~E.model.isin(["mean"])]
        E["se"] = (E.pred - E.y) ** 2
        # MSE relative to HAR per coin and split, then mean over coins
        m = E.groupby(["split", "dataset", "groupset", "model"]).se.mean()
        har = m.xs(("P", "har"), level=("groupset", "model"))
        rel = (m / har.reindex(m.index.droplevel(["groupset", "model"])).to_numpy()).groupby(
            level=["split", "groupset", "model"]).mean()
        val, test = rel.loc["val"], rel.loc["test"]
        nets = [k for k in val.index if not k[1].startswith(("har", "lgbm")) and k[1] not in
                ("persistence", "garch", "gjr", "ridge")]
        sel = min(nets, key=lambda k: val[k])
        best = min(nets, key=lambda k: test[k])
        rows.append(dict(freq=f, selected=f"{sel[1]} {sel[0]}", sel_val=val[sel], sel_test=test[sel],
                         test_best=f"{best[1]} {best[0]}", best_test=test[best]))
        print(f, rows[-1], flush=True)
        # MCS on the test losses averaged over coins at each timestamp
        T = E[E.split == "test"]
        L = T.pivot_table(index="time", columns=["groupset", "model"], values="se", aggfunc="mean").dropna(axis=1)
        # universe: networks and the HAR family; ridge (extreme forecasts on some coins), persistence and
        # GARCH are clearly dominated and would only inflate the bootstrap variance
        drop = {"persistence", "ridge", "garch", "gjr"}
        L = L.loc[:, [c for c in L.columns if c[1] not in drop and not (c[1] == "har" and c[0] != "P")]]
        L = L.T.drop_duplicates().T
        mcs = MCS(L.to_numpy(), size=0.10, reps=2000, block_size=max(2, int(len(L) ** (1 / 3))), seed=0)
        mcs.compute()
        inc = mcs.included
        names = [f"{L.columns[i][1]} {L.columns[i][0]}" for i in inc]
        mcs_rows.append(dict(freq=f, n_models=L.shape[1], n_mcs=len(inc), members="; ".join(names)))
        print(f, rows[-1], "\nMCS(90%):", len(inc), "of", L.shape[1], names, flush=True)
    pd.DataFrame(rows).to_csv(OUT_T / "review_selection.csv", index=False)
    pd.DataFrame(mcs_rows).to_csv(OUT_T / "review_mcs.csv", index=False)



def main_seeds():
    """Seed spread of the twin ratios (ratio of single-seed forecasts, seed k vs seed k)
    and seed-matched pooling gains (pooled ensembles cut to the single-coin seeds 0-2)."""
    from paper_tables_final import TWINS, run
    from src.paper.evaluate import dm_table
    rows, pool_rows = [], []
    for f in ("4h", "1h", "1d"):
        single, pooled = run(f"final_single_{f}"), run(f"final_pool_{f}")
        single, pooled = single[single.task == "range"], pooled[pooled.task == "range"]
        for tag, df in (("single", single), ("pooled", pooled)):
            sfx = POOL if tag == "pooled" else ""
            for q, r in TWINS:
                x = df[(df.groupset == "PMSE") & df.model.isin([q + sfx, r + sfx])]
                if x.model.nunique() < 2:
                    continue
                per = []
                for k in sorted(x.seed.unique()):
                    t = dm_table(x[x.seed == k], {"range": ("PMSE", r + sfx)})
                    per.append(t.rel_loss.mean())
                ens = dm_table(x, {"range": ("PMSE", r + sfx)}).rel_loss.mean()
                rows.append(dict(freq=f, setting=tag, pair=f"{q}/{r}", ensemble=ens, seed_min=min(per),
                                 seed_max=max(per), seed_sd=float(np.std(per, ddof=1)), n_seeds=len(per)))
        both = pd.concat([single, pooled[pooled.seed <= 2]])
        both = both[(both.groupset == "PMSE") | ((both.groupset == "P") & (both.model == "har"))]
        s = dm_table(both, {"range": ("P", "har")}).groupby("model").rel_loss.mean()
        for m in ("lstm", "qlstm", "transformer", "qtransformer", "hl_meta", "hq_meta", "ht_meta", "hqt_meta"):
            if m in s and m + POOL in s:
                pool_rows.append(dict(freq=f, model=m, single=s[m], pooled3=s[m + POOL], gain=1 - s[m + POOL] / s[m]))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_seed_spread.csv", index=False)
    p = pd.DataFrame(pool_rows)
    p.to_csv(OUT_T / "review_pooling_seedmatched.csv", index=False)
    print(t.round(4).to_string(index=False))
    print(p.round(4).to_string(index=False))



def main_altdata():
    """Value of the LunarCrush groups beyond exchange data: PMSE versus PM for every pooled
    model, all frequencies (coin-mean MSE ratio, coins better, cross-coin DM)."""
    from paper_tables_final import _pair, run
    rows = []
    for f in ("4h", "1h", "1d"):
        df = run(f"final_pool_{f}")
        df = df[df.task == "range"]
        for m in sorted(df.model.unique()):
            if ((df.groupset == "PM") & (df.model == m)).any() and ((df.groupset == "PMSE") & (df.model == m)).any():
                rows.append(dict(freq=f, model=m, **_pair(df, ("PMSE", m), ("PM", m))))
            if ((df.groupset == "P") & (df.model == m)).any() and ((df.groupset == "PM") & (df.model == m)).any():
                rows.append(dict(freq=f, model=m + " PM/P", **_pair(df, ("PM", m), ("P", m))))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_altdata.csv", index=False)
    print(t.round(4).to_string(index=False))



def _econ_inputs():
    """Saved 4h variance forecasts plus the equal-weight HAR + model combinations, and the
    no-trade-band volatility-targeting strategy shared by main_econ and main_econci."""
    from paper_tables_final import ECON, ECON_R8, level_har
    from src.paper.economic import BARS_PER_YEAR, load_preds, variance_forecasts
    n = BARS_PER_YEAR["4h"]
    tgt = 0.40 / np.sqrt(n)
    dirs = [run_dir(f"final_{x}_4h") for x in ("single", "pool", "harx", "har_season", "lgbm", "calendar_pool",
                                                "lgbmlags")]
    keep = lambda d: d[[(g, m) in ECON + ECON_R8 for g, m in zip(d.groupset, d.model)]]
    test = pd.concat([keep(load_preds(d)) for d in dirs if d])
    val = pd.concat([keep(load_preds(d, split="val")) for d in dirs if d])
    # round 8: plus HAR recalibrated to the target on validation data (paper_tables_final.level_har)
    V = level_har(variance_forecasts(test, val), variance_forecasts(val, val), "4h")
    # equal-weight combination of the HAR and network variance forecasts
    har = V[(V.groupset == "P") & (V.model == "har")].set_index(["dataset", "time"])
    combos = []
    for gs, m in (("PM", "hl_gattn-pool"), ("PMSE", "hq_meta-pool"), ("PMSE", "transformer-pool"),
                  ("PMSE", "hars"), ("PM", "lgbms-pool"),     # round 6: combinations without a network
                  ("PMC", "hl_gattn-pool"), ("PM", "lgbmsl-pool")):   # round 8
        g = V[(V.groupset == gs) & (V.model == m)].set_index(["dataset", "time"])
        idx = g.index.intersection(har.index)
        c = g.loc[idx].copy()
        c["var"] = 0.5 * (g.loc[idx, "var"] + har.loc[idx, "var"])
        c["model"] = "combo:" + m
        combos.append(c.reset_index())
    V = pd.concat([V] + combos, ignore_index=True)

    def strategy(g, band):
        w_star = np.minimum(2.0, tgt / np.sqrt(g["var"].to_numpy()))
        w = np.empty_like(w_star)
        prev = w_star[0]
        for t, x in enumerate(w_star):            # rebalance only outside the no-trade band
            prev = x if abs(x - prev) > band else prev
            w[t] = prev
        simple = np.expm1(g.r.to_numpy())
        turn = np.abs(np.diff(w, prepend=w[0]))
        return w * simple - 0.0005 * turn, turn

    return V, n, strategy


def main_econci(reps=1000, seed=0):
    """Round 6d: moving-block bootstrap 90% intervals for the cross-coin mean economic statistics
    against HAR (gamma = 10 and gamma = 1 fees, equal-volatility fee, net Sharpe difference), with
    and without the 0.25 no-trade band. The same resampled time blocks are applied to every coin
    and strategy, so the cross-coin and cross-model dependence is kept. Block length
    T^(1/3) as in the MCS; 4x that length as a robustness check."""
    from src.paper.economic import fko_fee
    V, n, strategy = _econ_inputs()
    rng = np.random.default_rng(seed)

    def stats(rp, b):
        s = np.std(b) / np.std(rp)
        return (fko_fee(rp, b, 10) * n * 1e4, fko_fee(rp, b, 1) * n * 1e4,
                fko_fee(rp * s, b, 10) * n * 1e4,
                (np.mean(rp) / np.std(rp) - np.mean(b) / np.std(b)) * np.sqrt(n))

    rows = []
    for band in (0.0, 0.25):
        S = {k: strategy(g.sort_values("time"), band)[0]
             for k, g in V.groupby(["dataset", "groupset", "model"])}
        T = min(len(v) for v in S.values())
        assert max(len(v) for v in S.values()) == T, "unequal test lengths across coins"
        coins = sorted({k[0] for k in S})
        models = sorted({k[1:] for k in S if k[1:] != ("P", "har")})
        for mult in (1, 4):
            L = mult * max(2, int(T ** (1 / 3)))
            nb = -(-T // L)
            draws = [(rng.integers(0, T - L + 1, nb)[:, None] + np.arange(L)).ravel()[:T]
                     for _ in range(reps)]
            for gs, m in models:
                pts = np.array([stats(S[(c, gs, m)], S[(c, "P", "har")]) for c in coins]).mean(0)
                bs = np.array([np.array([stats(S[(c, gs, m)][ix], S[(c, "P", "har")][ix])
                                         for c in coins]).mean(0) for ix in draws])
                lo, hi = np.percentile(bs, [5, 95], axis=0)
                for j, nm in enumerate(("fee10", "fee1", "fee_eq", "dsharpe")):
                    rows.append(dict(band=band, block=L, groupset=gs, model=m, stat=nm,
                                     point=pts[j], lo=lo[j], hi=hi[j]))
            print("econci band", band, "block", L, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_econ_ci.csv", index=False)
    print(t[t.block == t.block.min()].round(2).to_string())
    main_econci_tex()


def main_econci_tex():
    """LaTeX table of the bootstrap intervals for the five HAR combinations (short blocks)."""
    t = pd.read_csv(OUT_T / "review_econ_ci.csv")
    t = t[(t.block == t.block.min()) & (t.model.str.startswith("combo:") | (t.model == "har_lvl"))]
    names = {("PMSE", "combo:hq_meta-pool"): "HAR + \\HQ{} (meta, PMSE)",
             ("PM", "combo:hl_gattn-pool"): "HAR + H-LSTM (group attn., PM)",
             ("PMC", "combo:hl_gattn-pool"): "HAR + H-LSTM (group attn., PMC)",
             ("PMSE", "combo:transformer-pool"): "HAR + Transformer (PMSE)",
             ("PM", "combo:lgbms-pool"): "HAR + LightGBM-S (PM)",
             ("PM", "combo:lgbmsl-pool"): "HAR + LightGBM-S (lagged, PM)",
             ("PMSE", "combo:hars"): "HAR + HAR-S", ("P", "har_lvl"): "HAR (level-calibrated)"}
    cols = [(0.0, "fee10", 0), (0.0, "fee1", 0), (0.0, "fee_eq", 0), (0.0, "dsharpe", 2),
            (0.25, "fee10", 0), (0.25, "dsharpe", 2)]

    def cell(r, d):
        return f"${r.point:+.{d}f}$ \\scriptsize[${r.lo:+.{d}f}$, ${r.hi:+.{d}f}$]"
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{4}{c}{$\\delta = 0$} & \\multicolumn{2}{c}{$\\delta = 0.25$} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
             "Combination & Fee $\\gamma{=}10$ & Fee $\\gamma{=}1$ & Eq.\\ vol. & $\\Delta$SR & Fee $\\gamma{=}10$ & $\\Delta$SR \\\\",
             "\\midrule"]
    for (gs, m), nm in names.items():
        g = t[(t.groupset == gs) & (t.model == m)].set_index(["band", "stat"])
        lines.append(nm + " & " + " & ".join(cell(g.loc[(b, s)], d) for b, s, d in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex = "\n".join(lines).replace("$-0.00$", "$0.00$").replace("$+0.00$", "$0.00$")
    (OUT_T / "review_econ_ci.tex").write_text(tex + "\n")
    print("\n".join(lines))


def main_econ():
    """Economic add-ons from saved 4h forecasts: gamma = 1 fee, Gaussian VaR, a HAR + network
    forecast combination, a no-trade band against turnover, and the directional strategy."""
    from src.paper.economic import directional, fko_fee, var_backtest
    V, n, strategy = _econ_inputs()
    rows = []
    for band in (0.0, 0.1, 0.25):
        series = {}
        for (ds, gs, m), g in V.groupby(["dataset", "groupset", "model"]):
            rp, turn = strategy(g.sort_values("time"), band)
            series[(ds, gs, m)] = (rp, turn)
        for (ds, gs, m), (rp, turn) in series.items():
            b = series[(ds, "P", "har")][0]
            k = min(len(b), len(rp))
            rows.append(dict(band=band, dataset=ds, groupset=gs, model=m,
                             sharpe=np.mean(rp) / np.std(rp) * np.sqrt(n), turnover=np.mean(turn) * n,
                             vol_err=abs(np.std(rp) * np.sqrt(n) - 0.40),
                             fee10=fko_fee(rp[:k], b[:k], 10) * n * 1e4 if m != "har" else np.nan,
                             fee1=fko_fee(rp[:k], b[:k], 1) * n * 1e4 if m != "har" else np.nan))
    t = pd.DataFrame(rows)
    agg = t.groupby(["band", "groupset", "model"]).agg(sharpe=("sharpe", "mean"), turnover=("turnover", "mean"),
                                                       vol_err=("vol_err", "mean"), fee10=("fee10", "mean"),
                                                       fee1=("fee1", "mean"),
                                                       fee10_pos=("fee10", lambda v: int((v > 0).sum())))
    agg.to_csv(OUT_T / "review_econ.csv")
    print(agg.round(3).to_string())
    econ_bands_tex(agg)
    g = var_backtest(V, 0.01, "normal")
    g5 = var_backtest(V, 0.05, "normal")
    for a, v in (("1%", g), ("5%", g5)):
        v["rej"] = (v.p_uc < .05) | (v.p_ind < .05)
        print("Gaussian VaR", a, v.groupby(["groupset", "model"]).agg(rate=("rate", "mean"), rej=("rej", "sum")).round(4).to_string())
    d = directional(run_dir("final_return_4h"), "4h")
    dd = d.groupby(["groupset", "model"]).agg(sharpe=("sharpe", "mean"), ann_ret=("ann_ret", "mean"))
    dd.to_csv(OUT_T / "review_directional.csv")
    print(dd.round(3).to_string())



def econ_bands_tex(agg):
    """Round 5: compact table of the no-trade band results (Appendix Q) from review_econ.csv:
    annual turnover, net Sharpe ratio and gamma = 10 fee (bp/yr vs HAR with the same band)."""
    from paper_tables_final import ECON, label
    order = []
    for k in ECON:                                    # round 8: level-calibrated HAR, lagged LightGBM-S
        order += [k] + ([("P", "har_lvl")] if k == ("P", "har") else [("PM", "lgbmsl-pool")] if k == ("PM", "lgbms-pool") else [])
    order = [k for k in order if (0.0,) + k in agg.index]
    order += [k for k in [("PMSE", "combo:hq_meta-pool"), ("PM", "combo:hl_gattn-pool"),
                          ("PMC", "combo:hl_gattn-pool"), ("PMSE", "combo:transformer-pool"),
                          ("PMSE", "combo:hars"), ("PM", "combo:lgbms-pool"),
                          ("PM", "combo:lgbmsl-pool")] if (0.0,) + k in agg.index]
    with open(OUT_T / "r5_econ_bands.tex", "w") as fh:
        fh.write("% generated by experiments/paper_review_checks.py econ (no-trade bands, 4h, 5 bp costs)\n")
        fh.write("\\setlength{\\tabcolsep}{3pt}\n")
        fh.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n")
        fh.write(" & \\multicolumn{3}{c}{$\\delta = 0$} & \\multicolumn{3}{c}{$\\delta = 0.1$} & "
                 "\\multicolumn{3}{c}{$\\delta = 0.25$} \\\\\n")
        fh.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n")
        fh.write("Model, inputs & " + " & ".join(["Turn. & Sharpe & Fee"] * 3) + " \\\\\n\\midrule\n")
        for gs, m in order:
            name = ("50/50 HAR + " + label(gs, m[6:])) if m.startswith("combo:") else label(gs, m)
            cells = []
            for band in (0.0, 0.1, 0.25):
                r = agg.loc[(band, gs, m)]
                cells += [f"{r.turnover:.0f}", f"{r.sharpe:.2f}", "--" if pd.isna(r.fee10) else f"{r.fee10:.0f}"]
            fh.write(f"{name} & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


PAIRS_TEXT = [  # (frequency, a, b): every pairwise claim of Section 5.1, a vs b
    ("hl_meta", "PMSE", "lstm", "PMSE"), ("ht_meta", "PMSE", "transformer", "PMSE"),
    ("lstm", "PMSE", "lstm", "P"), ("transformer", "PMSE", "transformer", "P"),
    ("ht_meta", "PMSE", "transformer", "P"), ("hl_meta", "PMSE", "lstm", "P"),
    ("transformer-pool", "PMSE", "transformer", "PMSE"), ("transformer-pool", "PMSE", "transformer-pool", "P"),
    ("transformer-pool", "PM", "transformer-pool", "P"),
    ("hl_gattn-pool", "PM", "transformer-pool", "PM"), ("hl_gattn-pool", "PMSE", "transformer-pool", "PMSE"),
    ("hl_gattn-pool", "PMSE", "hl_gattn-pool", "PM"),
    ("hl_gattn-pool", "PM", "hars", "PMSE"), ("hl_gattn-pool", "PM", "harxs", "PMSE"),
    ("hl_gattn-pool", "PM", "harxs-pool", "PMSE"), ("ht_concat-pool", "PMSE", "hars", "PMSE"),
    ("ht_concat-pool", "PMSE", "harxs", "PMSE"), ("transformer-pool", "PM", "hars", "PMSE"),
    ("harx", "PMSE", "har", "P"), ("hars", "PMSE", "har", "P"), ("harxs", "PMSE", "hars", "PMSE"),
    ("hars-pool", "PMSE", "hars", "PMSE"), ("har-pool", "PMSE", "har", "P"),
]


def main_pairs():
    """Every pairwise comparison quoted in Section 5.1: coin-mean ratio, coins better,
    coins significant (DM, Holm over the 15 coins), cross-coin DM."""
    from paper_tables_final import _pair, setting
    from src.paper.evaluate import dm_table, holm
    rows = []
    for f in ("4h", "1h", "1d"):
        df = setting(f)
        have = set(zip(df.groupset, df.model))
        for a, ga, b, gb in PAIRS_TEXT:
            if (ga, a) not in have or (gb, b) not in have:
                continue
            x = df[[(g, m) in ((ga, a), (gb, b)) for g, m in zip(df.groupset, df.model)]]
            t = dm_table(x, {"range": (gb, b)})
            t["ph"] = holm(t.p)
            r = _pair(df, (ga, a), (gb, b))
            rows.append(dict(freq=f, a=f"{a} {ga}", b=f"{b} {gb}", ratio=r["ratio"], better=r["better"],
                             sig_better=int(((t.rel_loss < 1) & (t.ph < .05)).sum()),
                             sig_worse=int(((t.rel_loss > 1) & (t.ph < .05)).sum()), dm=r["dm"], p=r["p"]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_pairs.csv", index=False)
    print(t.round(4).to_string(index=False))



def main_calendar():
    """Pooled networks with the calendar group (seeds 0-2) against the same models without
    it, the seasonal HAR benchmarks, and the architecture / algebra pairs under PMC."""
    from paper_tables_final import _pair, _vs_har, run, setting
    rows, summ = [], []
    for f in ("4h", "1d", "1h"):
        cal = run(f"final_calendar_pool_{f}")
        if cal is None:
            continue
        cal2 = run(f"final_calendar2_pool_{f}")  # remaining hourly calendar networks (Deep Review)
        cal = pd.concat([cal, cal2]) if cal2 is not None else cal
        cal = cal[cal.task == "range"]
        n_fits = cal.groupby(["groupset", "model"]).apply(lambda g: len(g[["seed", "fold"]].drop_duplicates()))
        done = [k for k, v in n_fits.items() if v == 12]
        cal = cal[[(g, m) in done for g, m in zip(cal.groupset, cal.model)]]
        df = setting(f)
        base = df[(df.model.str.endswith(POOL) & (df.seed <= 2)) | df.model.isin(["hars", "harxs"])]
        x = pd.concat([base, cal])
        har = df[(df.groupset == "P") & (df.model == "har")]
        s = _vs_har(x, har)
        for (gs, m), r in s.iterrows():
            if gs in ("PC", "PMC") or m in ("hars", "harxs") or (gs in ("P", "PM") and (gs.replace("P", "P", 1), m) in
                                                                   [(g[:-1] if g.endswith("C") else g, mm) for g, mm in done]):
                summ.append(dict(freq=f, groupset=gs, model=m, mse=r.mse, better=r.better))
        for gs, m in done:
            ref = gs[:-1]                                   # PC -> P, PMC -> PM
            if ((x.groupset == ref) & (x.model == m)).any():
                rows.append(dict(freq=f, comp=f"{m} {gs} / {ref}", **_pair(x, (gs, m), (ref, m))))
            for b in ("hars", "harxs"):
                rows.append(dict(freq=f, comp=f"{m} {gs} / {b}", **_pair(x, (gs, m), ("PMSE", b))))
        for a, b in (("hl_gattn-pool", "lstm-pool"), ("ht_concat-pool", "transformer-pool"),
                     ("hq_gattn-pool", "hl_gattn-pool"), ("hqt_concat-pool", "ht_concat-pool")):
            for gs in ("PC", "PMC"):
                if (gs, a) in done and (gs, b) in done:
                    rows.append(dict(freq=f, comp=f"{a} / {b} {gs}", **_pair(x, (gs, a), (gs, b))))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_calendar.csv", index=False)
    u = pd.DataFrame(summ).sort_values(["freq", "mse"])
    u.to_csv(OUT_T / "review_calendar_summary.csv", index=False)
    print(u.round(4).to_string(index=False))
    print(t.round(4).to_string(index=False))
    # table: MSE/HAR without and with the calendar group, per frequency
    lab = {"lstm-pool": "LSTM", "transformer-pool": "Transformer", "hl_gattn-pool": "H-LSTM (group attn.)",
           "hq_gattn-pool": "H-QLSTM (group attn.)", "ht_concat-pool": "H-Transformer (concat)",
           "hqt_concat-pool": "H-QTransformer (concat)", "hars": "HAR-S", "harxs": "HAR-XS"}
    freqs = [f for f in ("4h", "1h", "1d") if f in set(u.freq)]
    names = {"4h": "4-hourly", "1h": "Hourly", "1d": "Daily"}
    get = lambda f, g, m: u[(u.freq == f) & (u.groupset == g) & (u.model == m)].mse
    with open(OUT_T / "review_calendar.tex", "w") as fh:
        fh.write("% generated by experiments/paper_review_checks.py calendar\n")
        fh.write("\\begin{tabular}{l" + "rr" * len(freqs) + "}\n\\toprule\n")
        fh.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{names[f]}}}" for f in freqs) + " \\\\\n")
        fh.write("Model (pooled) & " + " & ".join("PM & PMC" for _ in freqs) + " \\\\\n\\midrule\n")
        for m, l in lab.items():
            cells = []
            if m == "hars":
                fh.write("\\midrule\n")
            for f in freqs:
                if m in ("hars", "harxs"):  # seasonal benchmarks have calendar dummies: span PM and PMC
                    v = get(f, "PMSE", m)
                    cells.append(f"\\multicolumn{{2}}{{c}}{{{v.iloc[0]:.3f}}}" if len(v) else "\\multicolumn{2}{c}{--}")
                    continue
                for g in ("PM", "PMC"):
                    v = get(f, g, m)
                    cells.append(f"{v.iloc[0]:.3f}" if len(v) else "--")
            fh.write(f"{l} & " + " & ".join(cells) + " \\\\\n")
        # round 8: interacted HAR-S and the pooled LightGBM-S with calendar dummies, without and with
        # HAR-horizon averages of its inputs (experiments/paper_round8_checks.py bench_tests; lagged: seeds 0-2)
        bt = OUT_T / "r8_bench_tests.csv"
        if bt.exists():
            bt = pd.read_csv(bt)
            span = lambda v: f"\\multicolumn{{2}}{{c}}{{{v:.3f}}}" if pd.notna(v) else "\\multicolumn{2}{c}{--}"

            def bench(f, variant, a):
                r = bt[(bt.freq == f) & (bt.variant == variant) & (bt.a == a) & (bt.b == "har P")]
                return r.mean_ratio.iloc[0] if len(r) else np.nan

            def lgbms(f):
                sm = pd.read_csv(OUT_T / f"final_summary_{f}.csv")
                r = sm[(sm.groupset == "PM") & (sm.model == "lgbms-pool")]
                return r.mse.iloc[0] if len(r) else np.nan
            for l, fn in (("HAR-S$\\times$", lambda f: bench(f, "lgbm_lag_s0", "hars_x P")),
                          ("LightGBM-S, PM", lgbms),
                          ("LightGBM-S (lagged), PM", lambda f: bench(f, "lgbm_lag_3s", "lgbmsl-pool PM"))):
                fh.write(f"{l} & " + " & ".join(span(fn(f)) for f in freqs) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")



STEP = {"1h": pd.Timedelta("1h"), "4h": pd.Timedelta("4h"), "1d": pd.Timedelta("1D")}


def rv_variance(freq, ens):
    """Attach the realised variance of the target bar (sum of squared 1-minute log returns,
    data/binance_1m) and calibrate each model's variance forecast exp(2*pred) to it on the
    validation block, per coin, model and fold. Bars with < 90% of their minutes are dropped."""
    full = {"1h": 60, "4h": 240, "1d": 1440}[freq]
    out = []
    for ds, g in ens.groupby("dataset"):
        coin = ds.split("_")[0]
        path = f"data/binance_1m/{coin}_{freq}_rv.csv"
        if not os.path.exists(path):
            continue
        rv = pd.read_csv(path, index_col=0, parse_dates=True)
        rv = rv[rv.n_min >= 0.9 * full].rv
        g = g.assign(rv=rv.reindex(g.time + STEP[freq]).to_numpy())     # row t forecasts bar t+1
        g = g[np.isfinite(g.rv) & (g.rv > 0)]
        for key, h in g.groupby(["groupset", "model", "fold"]):
            v, t = h[h.split == "val"], h[h.split == "test"]
            if len(v) == 0 or len(t) == 0:
                continue
            c = np.mean(v.rv) / np.mean(np.exp(2 * v.pred))
            out.append(t.assign(h=c * np.exp(2 * t.pred)))
    V = pd.concat(out, ignore_index=True)
    V["ql"] = qlike_loss(V.rv, V.h)
    V["msev"] = (V.rv - V.h) ** 2
    V["mselog"] = (np.log(V.rv) - np.log(V.h)) ** 2
    return V


def main_rv():
    """Evaluation of the saved forecasts against 1-minute realised variance (review R2-4)."""
    rows, pairs = [], []
    for f in ("4h", "1h", "1d"):
        V = rv_variance(f, ensemble(load_both(RUNS[f] + [f"final_calendar_pool_{f}", f"final_calendar2_pool_{f}"]), seeds=[0, 1, 2]))
        base = V[(V.groupset == "P") & (V.model == "har")]
        for (gs, m), g in V.groupby(["groupset", "model"]):
            if (gs, m) == ("P", "har"):
                continue
            rec = dict(freq=f, groupset=gs, model=m)
            for col in ("ql", "msev", "mselog"):
                r = pair_ratio(V[((V.groupset == gs) & (V.model == m)) | ((V.groupset == "P") & (V.model == "har"))],
                               (gs, m), ("P", "har"), col)
                rec[col], rec[col + "_better"] = r["ratio"], r["better"]
                if col == "mselog":
                    rec["mselog_dm"], rec["mselog_p"] = r["dm"], r["p"]
            rows.append(rec)
        for name, (a, b) in PAIRS.items():
            if ((V.groupset == a[0]) & (V.model == a[1])).any() and ((V.groupset == b[0]) & (V.model == b[1])).any():
                for col in ("mselog", "ql"):
                    pairs.append(dict(freq=f, pair=name, loss=col, **pair_ratio(V, a, b, col)))
        print(f, flush=True)
    t = pd.DataFrame(rows).sort_values(["freq", "mselog"])
    t.to_csv(OUT_T / "review_rv.csv", index=False)
    p = pd.DataFrame(pairs)
    p.to_csv(OUT_T / "review_rv_pairs.csv", index=False)
    pd.set_option("display.width", 200)
    for f in ("4h", "1h", "1d"):
        print(t[t.freq == f].head(15).round(3).to_string(index=False))
    print(p.round(3).to_string(index=False))



FOLD_PAIRS = {  # (a, b): a/b < 1 means a is better
    "pooling (Transformer, PMSE)": (("PMSE", "transformer-pool"), ("PMSE", "transformer")),
    "pooling (H-LSTM, PMSE)": (("PMSE", "hl_meta-pool"), ("PMSE", "hl_meta")),
    "hierarchy, single (LSTM)": (("PMSE", "hl_meta"), ("PMSE", "lstm")),
    "hierarchy, single (Transformer)": (("PMSE", "ht_meta"), ("PMSE", "transformer")),
    "market data, pooled Transformer": (("PM", "transformer-pool"), ("P", "transformer-pool")),
    "social data, pooled Transformer": (("PMSE", "transformer-pool"), ("PM", "transformer-pool")),
    "best network vs HAR": (("PM", "hl_gattn-pool"), ("P", "har")),
    "best network vs HAR-S": (("PM", "hl_gattn-pool"), ("PMSE", "hars")),
    "QLSTM / LSTM, single": (("PMSE", "qlstm"), ("PMSE", "lstm")),
    "H-QLSTM / H-LSTM, pooled": (("PMSE", "hq_meta-pool"), ("PMSE", "hl_meta-pool")),
    "H-QT / H-T, pooled": (("PMSE", "hqt_meta-pool"), ("PMSE", "ht_meta-pool")),
}


BEST_NETWORK = {  # best network of each frequency's main grid (MSE, Tables 3-4 / S1-S3)
    "4h": ("PM", "hl_gattn-pool"), "1h": ("PM", "hl_gattn-pool"), "1d": ("PMSE", "ht_concat-pool")}


def main_folds():
    """Sub-period robustness: the key comparisons on each of the four 135-day test blocks
    (seed-matched, seeds 0-2), plus the market conditions of each block."""
    from paper_tables_final import setting
    from src.paper.evaluate import losses
    rows, cond = [], []
    for f in ("4h", "1h", "1d"):
        df = setting(f)
        df = df[df.seed <= 2]
        ens = (df.groupby(["dataset", "groupset", "model", "fold", "time"], as_index=False)
                 .agg(y=("y", "first"), pred=("pred", "mean")))
        ens["se"] = (ens.pred - ens.y) ** 2
        for name, (a, b) in FOLD_PAIRS.items():
            if name.startswith("best network"):
                a = BEST_NETWORK[f]
            A = ens[(ens.groupset == a[0]) & (ens.model == a[1])].set_index(["dataset", "fold", "time"]).se
            B = ens[(ens.groupset == b[0]) & (ens.model == b[1])].set_index(["dataset", "fold", "time"]).se
            idx = A.index.intersection(B.index)
            if len(idx) == 0:
                continue
            r = (A.loc[idx].groupby(level=[0, 1]).mean() / B.loc[idx].groupby(level=[0, 1]).mean())
            per = r.groupby(level=1).agg(["mean", lambda v: int((v < 1).sum())])
            for k, (m, nb) in per.iterrows():
                rows.append(dict(freq=f, comparison=name, fold=int(k), ratio=m, better=nb))
        if f == "4h":
            har = ens[(ens.groupset == "P") & (ens.model == "har")]
            for k, g in har.groupby("fold"):
                cond.append(dict(fold=int(k), start=g.time.min().date(), end=g.time.max().date(),
                                 mean_log_range=g.y.mean()))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_folds.csv", index=False)
    piv = t.pivot_table(index=["freq", "comparison"], columns="fold", values="ratio")
    pd.set_option("display.width", 200)
    print(piv.round(3).to_string())
    print(pd.DataFrame(cond).to_string(index=False))



def main_harrv():
    """HAR models fitted directly on 1-minute realised variance (log RV), with and without
    calendar dummies, compared with the networks on the log-RV MSE of review_rv.csv.
    The RV history starts in October 2024, so each fold's HAR-RV is estimated on all RV
    observations before its test block (6 to 24 months)."""
    from src.paper.baselines import HAR_LAGS, _calendar, _har_design, _ols
    from paper_tables_final import setting
    rows = []
    for f in ("4h", "1h", "1d"):
        full = {"1h": 60, "4h": 240, "1d": 1440}[f]
        # test blocks and the HAR (range) benchmark's calibrated loss from the saved forecasts
        V = rv_variance(f, ensemble(load_both([f"final_single_{f}"]), seeds=[0]))
        V = V[(V.groupset == "P") & (V.model == "har")]
        for ds, g in V.groupby("dataset"):
            coin = ds.split("_")[0]
            rv = pd.read_csv(f"data/binance_1m/{coin}_{f}_rv.csv", index_col=0, parse_dates=True)
            ok = (rv.n_min >= 0.9 * full) & (rv.rv > 0)
            lrv = np.log(rv.rv.where(ok)).ffill()
            X = _har_design(lrv.to_numpy(), HAR_LAGS[f])
            C = _calendar(lrv.index, STEP[f])                     # calendar of bar t+1
            y = lrv.shift(-1).to_numpy()                          # log RV of bar t+1
            idx = pd.Series(np.arange(len(lrv)), index=lrv.index)
            for fold, h in g.groupby("fold"):
                t0 = h.time.min()
                tr = idx[idx.index < t0].to_numpy()
                tr = tr[~np.isnan(X[tr]).any(1) & np.isfinite(y[tr])]
                te = idx.reindex(h.time).to_numpy()
                for name, D in (("har_rv", X), ("har_rv_s", np.column_stack([X, C]))):
                    b = _ols(D[tr], y[tr])
                    pred = b[0] + D[te] @ b[1:]
                    target = np.log(h.rv.to_numpy())
                    rows.append(dict(freq=f, dataset=ds, fold=fold, model=name,
                                     mselog=np.mean((target - pred) ** 2),
                                     har_mselog=np.mean((target - np.log(h.h.to_numpy())) ** 2), n=len(h)))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    agg = (t.groupby(["freq", "model", "dataset"]).apply(lambda g: np.average(g.mselog, weights=g.n) /
                                                          np.average(g.har_mselog, weights=g.n))
             .groupby(level=[0, 1]).agg(["mean", lambda v: int((v < 1).sum())]))
    agg.columns = ["mselog_vs_har", "better"]
    agg.to_csv(OUT_T / "review_harrv.csv")
    print(agg.round(3).to_string())



def main_harmonth():
    """Round 7: intraday HAR and HAR-S with an added monthly (30-day) lag, refitted on the same
    folds and test bars as the saved benchmarks; per-coin MSE ratios against the standard
    benchmarks, and the best networks measured against the extended benchmarks."""
    import yaml
    from src.paper.baselines import HAR_LAGS, _calendar, _har_design, _ols
    from src.paper.data import make_dataset, walk_forward_folds
    from paper_tables_final import run, setting
    month = {"4h": 180, "1h": 720}
    nets = [("PM", "hl_gattn-pool"), ("PMSE", "hq_meta-pool"), ("PM", "lgbms-pool")]
    rows = []
    for f in ("4h", "1h"):
        cfg = yaml.safe_load(open(f"configs/paper/final/har_season_{f}.yaml"))
        fc, win = cfg["folds"], cfg["window"][f]
        df = setting(f)
        cal = run(f"final_calendar_pool_{f}")
        df = pd.concat([df, cal[(cal.task == "range") & (cal.groupset == "PMC") & (cal.model == "hl_gattn-pool")]])
        df = df[df.seed == 0]
        for ds in cfg["datasets"]:
            coin = ds.split("_")[0]
            d = make_dataset(coin, f, start=cfg.get("start"))
            fr, y = d.frame, d.frame["y_range"].to_numpy(float)
            folds = walk_forward_folds(fr.index, fc["n"], fc["test_days"], fc["val_days"], end=fc.get("end"))
            C = _calendar(fr.index, d.step)
            pred = {}
            for name, lags, cs in (("har", HAR_LAGS[f], False), ("har_m", HAR_LAGS[f] + (month[f],), False),
                                   ("hars", HAR_LAGS[f], True), ("hars_m", HAR_LAGS[f] + (month[f],), True)):
                H = _har_design(fr["log_range"].to_numpy(float), lags)
                if cs:
                    H = np.column_stack([H, C])
                out = pd.Series(np.nan, index=fr.index)
                for fo in folds:
                    tr, te = fo["train"], fo["test"]
                    te = te[te >= win - 1]
                    ok = tr[~np.isnan(H[tr]).any(1)]
                    b = _ols(H[ok], y[ok])
                    out.iloc[te] = b[0] + H[te] @ b[1:]
                pred[name] = out
            g = df[df.dataset == ds]
            times = g[(g.groupset == "P") & (g.model == "har")].set_index("time")
            P = pd.DataFrame({k: v.reindex(times.index).to_numpy() for k, v in pred.items()}, index=times.index)
            P["y"] = times.y
            P["har_saved"] = times.pred
            for gs, m in nets + [("PMC", "hl_gattn-pool")]:
                P[m + "|" + gs] = g[(g.groupset == gs) & (g.model == m)].set_index("time").pred.reindex(P.index)
            mse = {c: np.mean((P[c] - P.y) ** 2) for c in P.columns if c != "y"}
            rows.append(dict(freq=f, dataset=ds, **mse))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_harmonth_raw.csv", index=False)
    out = []
    for f, g in t.groupby("freq"):
        def r(a, b):
            v = g[a] / g[b]
            return dict(freq=f, a=a, b=b, mean=v.mean(), better=int((v < 1).sum()))
        out += [r("har", "har_saved"), r("har_m", "har"), r("hars_m", "hars")]
        for c in [c for c in g.columns if "|" in c]:
            out += [r(c, "har"), r(c, "har_m"), r(c, "hars"), r(c, "hars_m")]
    o = pd.DataFrame(out)
    o.to_csv(OUT_T / "review_harmonth.csv", index=False)
    print(o.round(4).to_string(index=False))


def cross_losses(df, a, b, scale=None):
    """Seed-averaged squared-error losses of a and b on calendar-aligned dates, averaged over
    coins on each date. With `scale` (dataset -> HAR mean test MSE) each coin's losses are
    divided by that coin's HAR MSE first, so every coin gets equal weight (round 5)."""
    from src.paper.evaluate import seed_mean, losses
    ens = seed_mean(df)
    la, lb = [], []
    for ds, g in ens.groupby("dataset"):
        A = g[(g.groupset == a[0]) & (g.model == a[1])].set_index("time")
        B = g[(g.groupset == b[0]) & (g.model == b[1])].set_index("time")
        t = A.index.intersection(B.index)
        if len(t) == 0:
            continue
        w = 1.0 if scale is None else 1.0 / scale[ds]
        la.append(pd.Series(w * losses("range", A.loc[t, "y"].to_numpy(), A.loc[t, "pred"].to_numpy())["se"], index=t))
        lb.append(pd.Series(w * losses("range", B.loc[t, "y"].to_numpy(), B.loc[t, "pred"].to_numpy())["se"], index=t))
    LA = pd.concat(la, axis=1).mean(axis=1).sort_index()
    LB = pd.concat(lb, axis=1).mean(axis=1).reindex(LA.index)
    return LA, LB


def equivalence(df, a, b, margin=0.01, scale=None):
    """Cross-coin loss ratio of a over b with a 90% Newey-West confidence interval
    (pooled loss = mean over coins on each date). Equivalent at +-margin when the whole
    90% interval lies inside [1 - margin, 1 + margin] (two one-sided tests at 5%).
    `scale`: optional per-coin normalisation, see cross_losses."""
    LA, LB = cross_losses(df, a, b, scale)
    d = (LA - LB).to_numpy()
    n = len(d)
    lag = int(np.floor(n ** (1 / 3)))
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    se = np.sqrt(var / n) / LB.mean()
    r = 1 + d.mean() / LB.mean()
    lo, hi = r - 1.645 * se, r + 1.645 * se
    return dict(ratio=r, ci_lo=lo, ci_hi=hi, equivalent=bool(lo > 1 - margin and hi < 1 + margin))


def main_equiv():
    """Price-only twins and 90% equivalence intervals for the twin and fusion claims."""
    from paper_tables_final import TWINS, run, setting
    rows = []
    for f in ("4h", "1h", "1d"):
        df = setting(f)
        for tag, sfx in (("single", ""), ("pooled", POOL)):
            for gs in ("P", "PMSE"):
                for q, r in TWINS:
                    a, b = (gs, q + sfx), (gs, r + sfx)
                    x = df[[(g, m) in (a, b) for g, m in zip(df.groupset, df.model)]]
                    if x.model.nunique() < 2:
                        continue
                    rows.append(dict(freq=f, setting=tag, inputs=gs, pair=f"{q}/{r}", **equivalence(x, a, b)))
        print(f, flush=True)
    fnl = run("final_fusion_nl_pool_4h")
    pool = run("final_pool_4h")
    pool = pool[(pool.task == "range") & (pool.seed <= 2) & (pool.groupset == "PMSE")]
    x = pd.concat([pool, fnl[fnl.task == "range"]])
    for enc in ("hq", "hl"):
        for fu in ("phmn", "densen"):
            rows.append(dict(freq="4h", setting="pooled fusion", inputs="PMSE", pair=f"{enc}_{fu}/{enc}_metan",
                             **equivalence(x, ("PMSE", f"{enc}_{fu}-pool"), ("PMSE", f"{enc}_metan-pool"))))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_equivalence.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))


def main_controls():
    """Round-3 controls (three seeds, seed-matched to the main grid cut to seeds 0-2):
    weight-decayed flat single-coin networks, random 4x4 feature partitions for the
    hierarchical encoders, and the 2k-budget bracketing of the H-LSTM (H=8, 2,353 params).
    Writes review_controls.csv/.tex."""
    import paper_tables_final as T
    rows = []
    for f in ("4h", "1d"):
        base = T.setting(f)
        har = base[(base.groupset == "P") & (base.model == "har")]
        ref = base[base.seed.isin([0, 1, 2]) & (base.groupset == "PMSE")]
        v0 = T._vs_har(ref[ref.model.isin(["lstm", "transformer", "hl_meta", "ht_meta"])], har)
        for m in ("lstm", "transformer"):
            rows.append(dict(block="Weight decay (single coin)", freq=f, model=m, variant="none",
                             mse=v0.mse[("PMSE", m)]))
            for w in ("1e4", "1e3"):
                r = T.run(f"final_wd{w}_single_{f}")
                r = r[(r.task == "range") & (r.groupset == "PMSE") & (r.model == m)]
                rows.append(dict(block="Weight decay (single coin)", freq=f, model=m, variant=f"wd {w.replace('e', 'e-')}",
                                 mse=T._vs_har(r, har).mse[("PMSE", m)]))
        for m in ("hl_meta", "ht_meta"):
            rows.append(dict(block="Weight decay (single coin)", freq=f, model=m, variant="none", mse=v0.mse[("PMSE", m)]))
    base = T.setting("4h")
    har = base[(base.groupset == "P") & (base.model == "har")]
    for name, pooled in (("final_randgroups_pool_4h", True), ("final_randgroups_single_4h", False)):
        r = T.run(name)
        r = r[r.task == "range"]
        for m in sorted(r.model.unique()):
            sem = base[base.seed.isin([0, 1, 2]) & (base.groupset == "PMSE") & (base.model == m)]
            both = pd.concat([sem, r[r.model == m]])
            v = T._vs_har(both, har)
            rows.append(dict(block="Random partitions (4h)", freq="4h", model=m, variant="semantic", mse=v.mse[("PMSE", m)]))
            for g in ("R1", "R2", "R3"):
                x = T._pair(both, (g, m), ("PMSE", m))
                rows.append(dict(block="Random partitions (4h)", freq="4h", model=m, variant=g, mse=v.mse[(g, m)],
                                 ratio=x["ratio"], better=x["better"], p=x["p"]))
    b2 = T.run("final_budget_2000_pool_4h")
    b2 = b2[(b2.task == "range") & (b2.groupset == "PMSE")]
    b8 = T.run("final_budget_2350_pool_4h")
    b8 = b8[b8.task == "range"].assign(model="hl_meta8-pool")
    both = pd.concat([b2, b8])
    for a, lab in (("hl_meta-pool", "H-LSTM H=7 (1,891)"), ("hl_meta8-pool", "H-LSTM H=8 (2,353)")):
        x = T._pair(both, ("PMSE", "hq_meta-pool"), ("PMSE", a))
        rows.append(dict(block="2k bracketing (pooled 4h)", freq="4h", model="hq_meta-pool", variant=f"vs {lab}",
                         ratio=x["ratio"], better=x["better"], p=x["p"]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_controls.csv", index=False)
    print(t.round(4).to_string(), flush=True)


def main_matched():
    """Confound control for the Transformer twins: real Transformers retrained with GELU, two heads
    (flat) and a free width (final_twinmatch_pool_{freq}, pooled PMSE, seeds 0-2) against the
    quaternion models of the main pooled grid cut to the same seeds. Writes review_matched.csv."""
    import paper_tables_final as T
    rows = []
    for f in ("4h", "1h"):
        pool = T.run(f"final_pool_{f}")
        pool = pool[(pool.task == "range") & pool.seed.isin([0, 1, 2]) & (pool.groupset == "PMSE")]
        tm = T.run(f"final_twinmatch_pool_{f}")
        x = pd.concat([pool, tm[tm.task == "range"]])
        for q, r, r0 in (("qtransformer", "transformer_g", "transformer"), ("hqt_meta", "htg_meta", "ht_meta"),
                         ("hqt_concat", "htg_concat", "ht_concat")):
            a, b, b0 = ("PMSE", q + POOL), ("PMSE", r + POOL), ("PMSE", r0 + POOL)
            m = T._pair(x, a, b)
            o = T._pair(x, a, b0)
            rows.append(dict(freq=f, pair=f"{q}/{r}", ratio=m["ratio"], better=m["better"], n=m["n"], dm=m["dm"],
                             p=m["p"], ratio_orig=o["ratio"], p_orig=o["p"], **{f"eq_{k}": v for k, v in
                                                                                 equivalence(x, a, b).items()}))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_matched.csv", index=False)
    print(t.round(4).to_string(index=False), flush=True)


def main_mcs_calendar():
    """90% MCS (settings as in `select`) on seed-matched forecasts (networks cut to seeds 0-2)
    that include the pooled calendar networks (PC/PMC) of final_calendar_pool_{freq}."""
    from arch.bootstrap import MCS
    rows = []
    for f in ("4h", "1h", "1d"):
        E = ensemble(load_both(RUNS[f] + [f"final_calendar_pool_{f}", f"final_calendar2_pool_{f}"]), seeds=[0, 1, 2])
        E = E[(E.split == "test") & ~E.model.isin(["mean"])]
        E["se"] = (E.pred - E.y) ** 2
        L = E.pivot_table(index="time", columns=["groupset", "model"], values="se", aggfunc="mean").dropna(axis=1)
        drop = {"persistence", "ridge", "garch", "gjr"}
        L = L.loc[:, [c for c in L.columns if c[1] not in drop and not (c[1] == "har" and c[0] != "P")]]
        L = L.T.drop_duplicates().T
        mcs = MCS(L.to_numpy(), size=0.10, reps=2000, block_size=max(2, int(len(L) ** (1 / 3))), seed=0)
        mcs.compute()
        names = [f"{L.columns[i][1]} {L.columns[i][0]}" for i in mcs.included]
        n_cal = sum(c[0] in ("PC", "PMC") for c in L.columns)
        rows.append(dict(freq=f, n_models=L.shape[1], n_calendar_nets=n_cal, n_mcs=len(names), members="; ".join(names)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT_T / "review_mcs_calendar.csv", index=False)


LOSS_ROWS = [  # (label, groupset, model); seed-matched ensembles (seeds 0-2)
    ("HAR", "P", "har"), ("HAR-S", "PMSE", "hars"), ("LightGBM-S (pooled, PM)", "PM", "lgbms-pool"),
    ("H-LSTM (meta, single, PMSE)", "PMSE", "hl_meta"), ("Transformer (pooled, PMSE)", "PMSE", "transformer-pool"),
    ("H-Transformer (meta, pooled, PMSE)", "PMSE", "ht_meta-pool"),
    ("H-Transformer (concat, pooled, PMSE)", "PMSE", "ht_concat-pool"),
    ("H-LSTM (group attn., pooled, PM)", "PM", "hl_gattn-pool"),
    ("H-LSTM (group attn., pooled, PMC)", "PMC", "hl_gattn-pool"),
    ("H-Transformer (concat, pooled, PMC)", "PMC", "ht_concat-pool")]


def main_loss():
    """Compact loss-robustness table: calibrated QLIKE on the squared range, and log-MSE and
    calibrated QLIKE against 1-minute realised variance (from review_rv.csv / review_harrv.csv),
    all relative to HAR (mean over coins of per-coin loss ratios), seeds 0-2, with the number of
    coins (of 15) on which the model beats HAR in each column (round 5)."""
    hrv = pd.read_csv(OUT_T / "review_harrv.csv")
    rows = []
    core = {(gs, m) for _, gs, m in LOSS_ROWS}
    cols = ("ql_range", "rv_mselog", "rv_ql")
    for f in ("4h", "1h", "1d"):
        E = ensemble(load_both(RUNS[f] + [f"final_calendar_pool_{f}", f"final_calendar2_pool_{f}"]), seeds=[0, 1, 2])
        E = E[[(g, m) in core for g, m in zip(E.groupset, E.model)]]
        V = calibrated_variance(E)
        V["ql"] = qlike_loss(V.s2, V.h)
        R = rv_variance(f, E)
        for lab, gs, m in LOSS_ROWS:
            rec = dict(freq=f, label=lab)
            if (gs, m) == ("P", "har"):
                rec.update(ql_range=1.0, rv_mselog=1.0, rv_ql=1.0)
            elif ((V.groupset == gs) & (V.model == m)).any():
                for col, src, lc in zip(cols, (V, R, R), ("ql", "mselog", "ql")):
                    r = pair_ratio(src, (gs, m), ("P", "har"), lc)
                    rec[col], rec[col + "_better"] = r["ratio"], r["better"]
                    if col == "rv_mselog":   # DM only on the log-scale loss (QLIKE is too heavy-tailed)
                        rec["rv_mselog_dm"], rec["rv_mselog_p"] = r["dm"], r["p"]
                    else:                    # round 8: QLIKE markers from the Wilcoxon + block-bootstrap rule
                        q = ql_test(src, (gs, m), ("P", "har"), lc)
                        rec[col + "_pw"], rec[col + "_pb"], rec[col + "_p"] = q["p_wilcoxon"], q["p_boot"], q["p"]
            rows.append(rec)
        for lab, m in (("HAR-RV", "har_rv"), ("HAR-RV-S", "har_rv_s")):
            r = hrv[(hrv.freq == f) & (hrv.model == m)]
            rows.append(dict(freq=f, label=lab, rv_mselog=r.mselog_vs_har.iloc[0], rv_mselog_better=r.better.iloc[0]))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "review_loss.csv", index=False)
    print(t.round(3).to_string(), flush=True)
    loss_tex(t)


def ql_test(V, a, b, col="ql", reps=2000, seed=0):
    """Round 8: the two-part QLIKE rule of Section 5.4 for a against b. (i) Wilcoxon signed-rank test over
    coins of the per-coin log ratio of mean losses; (ii) moving-block bootstrap over calendar time (the
    same blocks for every coin, length floor(T^(1/3))) of the cross-coin mean of that log ratio, two-sided,
    equal-tailed (basic bootstrap). p = max of the two, so p < 0.05 requires both tests to reject."""
    from scipy import stats
    A = V[(V.groupset == a[0]) & (V.model == a[1])].set_index(["dataset", "time"])[col]
    B = V[(V.groupset == b[0]) & (V.model == b[1])].set_index(["dataset", "time"])[col]
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx].unstack(0).sort_index(), B.loc[idx].unstack(0).sort_index()
    a_, b_ = A.to_numpy(), B.to_numpy()
    lr = np.log(np.nanmean(a_, 0) / np.nanmean(b_, 0))
    pw = stats.wilcoxon(lr).pvalue
    T = len(a_)
    L = max(2, int(T ** (1 / 3)))
    rng = np.random.default_rng(seed)
    stat = lr.mean()
    boot = np.empty(reps)
    for k in range(reps):
        ix = (rng.integers(0, T - L + 1, -(-T // L))[:, None] + np.arange(L)).ravel()[:T]
        boot[k] = np.mean(np.log(np.nanmean(a_[ix], 0) / np.nanmean(b_[ix], 0)))
    c = boot - stat
    pb = min(1.0, 2 * min(float(np.mean(c <= stat)), float(np.mean(c >= stat))))
    return dict(p_wilcoxon=float(pw), p_boot=pb, p=max(float(pw), pb))


def loss_tex(t):
    """review_loss.tex from the review_loss.csv frame; coin counts in small type, tight columns.
    MSE_RV cells carry cross-coin DM markers against HAR (*: p < 0.05; dagger: Holm-adjusted
    over all MSE_RV tests in the table)."""
    from src.paper.evaluate import holm
    cols = ("ql_range", "rv_mselog", "rv_ql")
    t = t.copy()
    if "rv_mselog_p" in t:
        ok = t.rv_mselog_p.notna()
        t["rv_mselog_ph"] = np.nan
        t.loc[ok, "rv_mselog_ph"] = holm(t.loc[ok, "rv_mselog_p"].to_numpy())
    if "ql_range_p" in t:                     # round 8: Holm over all QLIKE tests (both columns) in the table
        ql = [(i, c) for c in ("ql_range", "rv_ql") for i in t.index if pd.notna(t.loc[i, c + "_p"])]
        adj = holm(np.array([t.loc[i, c + "_p"] for i, c in ql]))
        for c in ("ql_range", "rv_ql"):
            t[c + "_ph"] = np.nan
        for (i, c), v in zip(ql, adj):
            t.loc[i, c + "_ph"] = v
    order = [r[0] for r in LOSS_ROWS[:2]] + ["HAR-RV", "HAR-RV-S"] + [r[0] for r in LOSS_ROWS[2:]]

    def fmt(r, col):
        v, b = r.get(col), r.get(col + "_better")
        if pd.isna(v):
            return "--"
        p = r.get(col + "_p")
        star = "" if p is None or pd.isna(p) else _stars(p, r[col + "_ph"])
        return f"{v:.3f}{star}" if pd.isna(b) else f"{v:.3f}{star}{{\\scriptsize\\,({int(b)})}}"
    with open(OUT_T / "review_loss.tex", "w") as fh:
        fh.write("% generated by experiments/paper_review_checks.py loss; in parentheses: coins (of 15) better than HAR; MSE_RV: DM markers\n")
        fh.write("\\setlength{\\tabcolsep}{3pt}\n")
        fh.write("\\begin{tabular}{l" + "rrr" * 3 + "}\n\\toprule\n")
        fh.write(" & \\multicolumn{3}{c}{4-hourly} & \\multicolumn{3}{c}{Hourly} & \\multicolumn{3}{c}{Daily} \\\\\n")
        fh.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n")
        fh.write("Model & " + " & ".join(["QL$_R$ & MSE$_{RV}$ & QL$_{RV}$"] * 3) + " \\\\\n\\midrule\n")
        for lab in order:
            cells = []
            for f in ("4h", "1h", "1d"):
                r = t[(t.freq == f) & (t.label == lab)].iloc[0]
                cells += [fmt(r, c) for c in cols]
            fh.write(f"{lab} & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def _stars(p, ph):
    """*: unadjusted p < 0.05; dagger: Holm-adjusted p < 0.05 within the table."""
    s = ("*" if p < 0.05 else "") + ("\\dagger" if ph < 0.05 else "")
    return f"$^{{{s}}}$" if s else ""


def main_tables_holm():
    """Write review_altdata.tex and review_controls.tex from their CSVs, with Holm adjustment
    over every test shown in each table (family = table)."""
    from src.paper.evaluate import holm
    # value of the data groups
    t = pd.read_csv(OUT_T / "review_altdata.csv")
    lab = {"lstm-pool": "LSTM", "qlstm-pool": "QLSTM", "transformer-pool": "Transformer",
           "qtransformer-pool": "QTransformer", "hl_gattn-pool": "H-LSTM (group attn.)",
           "hq_gattn-pool": "H-QLSTM (group attn.)", "ht_concat-pool": "H-Transformer (concat)",
           "hqt_concat-pool": "H-QTransformer (concat)"}
    cells = {}
    for r in t.itertuples():
        m, kind = (r.model[:-5], "PM/P") if r.model.endswith(" PM/P") else (r.model, "PMSE/PM")
        if m in lab:
            cells[(m, r.freq, kind)] = r
    keys = list(cells)
    ph = dict(zip(keys, holm(np.array([cells[k].p for k in keys]))))
    with open(OUT_T / "review_altdata.tex", "w") as fh:
        fh.write("% generated by experiments/paper_review_checks.py tables_holm (from review_altdata.csv)\n")
        fh.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        fh.write(" & \\multicolumn{2}{c}{4-hourly} & \\multicolumn{2}{c}{Hourly} & \\multicolumn{2}{c}{Daily} \\\\\n")
        fh.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n")
        fh.write("Pooled model & PM/P & PMSE/PM & PM/P & PMSE/PM & PM/P & PMSE/PM \\\\\n\\midrule\n")
        for m, l in lab.items():
            row = []
            for f in ("4h", "1h", "1d"):
                for kind in ("PM/P", "PMSE/PM"):
                    r = cells.get((m, f, kind))
                    row.append("--" if r is None else f"{r.ratio:.3f}{_stars(r.p, ph[(m, f, kind)])} ({r.better})")
            fh.write(f"{l} & " + " & ".join(row) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    n = len(keys); print("altdata:", n, "tests,", sum(cells[k].p < .05 for k in keys), "unadj.,",
                         sum(ph[k] < .05 for k in keys), "Holm", flush=True)
    for k in keys:
        if cells[k].p < .05 and ph[k] >= .05:
            print("  loses:", k, round(cells[k].ratio, 4), round(cells[k].p, 4))
    # regularisation and grouping controls
    c = pd.read_csv(OUT_T / "review_controls.csv")
    rp = c[c.block.str.startswith("Random")].dropna(subset=["p"]).copy()
    rp["p_holm"] = holm(rp.p.to_numpy())
    print("controls (random partitions):", len(rp), "tests,", int((rp.p < .05).sum()), "unadj.,",
          int((rp.p_holm < .05).sum()), "Holm", flush=True)
    print(rp[(rp.p < .05) & (rp.p_holm >= .05)][["model", "variant", "ratio", "p"]].to_string(), flush=True)
    wd = c[c.block.str.startswith("Weight")]
    g = lambda f, m, v: wd[(wd.freq == f) & (wd.model == m) & (wd.variant == v)].mse
    fmt = lambda x: f"{x.iloc[0]:.3f}" if len(x) else ""
    names = {"lstm": "LSTM", "transformer": "Transformer", "hl_meta": "H-LSTM (meta)", "ht_meta": "H-Transformer (meta)"}
    with open(OUT_T / "review_controls.tex", "w") as fh:
        fh.write("% generated by experiments/paper_review_checks.py tables_holm (from review_controls.csv)\n")
        fh.write("\\begin{tabular}{llrrr}\n\\toprule\n")
        fh.write("\\multicolumn{5}{l}{\\emph{(a) Weight decay, single-coin models, PMSE inputs: MSE/HAR}} \\\\\n")
        fh.write(" & & none & $10^{-4}$ & $10^{-3}$ \\\\\n\\midrule\n")
        for f, fl in (("4h", "4-hourly"), ("1d", "Daily")):
            for i, m in enumerate(names):
                fh.write(f"{fl if i == 0 else ''} & {names[m]} & {fmt(g(f, m, 'none'))} & {fmt(g(f, m, 'wd 1e-4'))} & "
                         f"{fmt(g(f, m, 'wd 1e-3'))} \\\\\n")
        fh.write("\\midrule\n\\multicolumn{5}{l}{\\emph{(b) Random partitions, 4-hourly, PMSE: MSE ratio to the semantic grouping}} \\\\\n")
        fh.write(" & & R1 & R2 & R3 \\\\\n\\midrule\n")
        rl = {"hl_meta-pool": ("Pooled", "H-LSTM (meta)"), "hl_gattn-pool": ("", "H-LSTM (group attn.)"),
              "hq_meta-pool": ("", "\\HQ{} (meta)"), "hl_meta": ("Single coin", "H-LSTM (meta)")}
        for m, (blk, l) in rl.items():
            cells = []
            for v in ("R1", "R2", "R3"):
                r = rp[(rp.model == m) & (rp.variant == v)].iloc[0]
                cells.append(f"{r.ratio:.3f}{_stars(r.p, r.p_holm)}")
            fh.write(f"{blk} & {l} & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


# ============================================================================= round 5
def har_scale(df):
    """Per-coin HAR mean test MSE of the log range: the normaliser of the round-5 cross-coin tests."""
    h = df[(df.groupset == "P") & (df.model == "har")]
    return ((h.pred - h.y) ** 2).groupby(h.dataset).mean()


def main_norm_tests():
    """Round 5: the cross-coin DM tests of the twin table (Table 8) and the +-1% equivalence
    intervals (Fig. 2) with each coin's losses divided by that coin's HAR mean test MSE before
    the cross-sectional average, next to the unnormalised versions used in the paper.
    Writes r5_norm_twins.csv and r5_norm_equiv.csv."""
    from paper_tables_final import LABEL, TWINS, run, setting
    from src.paper.evaluate import holm
    twin_rows, eq_rows = [], []
    for f in ("4h", "1h", "1d"):
        df = setting(f)
        scale = har_scale(df)
        for tag, sfx in (("single", ""), ("pooled", POOL)):
            for gs in ("P", "PMSE"):
                for q, r in TWINS:
                    a, b = (gs, q + sfx), (gs, r + sfx)
                    x = df[[(g, m) in (a, b) for g, m in zip(df.groupset, df.model)]]
                    if x.model.nunique() < 2:
                        continue
                    if gs == "PMSE":                                 # the rows of Table 8
                        rec = dict(setting=f"{tag} {f}", pair=f"{LABEL[q]} / {LABEL[r]}")
                        for kind, sc in (("raw", None), ("norm", scale)):
                            LA, LB = cross_losses(x, a, b, sc)
                            st, p = diebold_mariano((LA - LB).to_numpy(), np.zeros(len(LA)))
                            rec.update({f"ratio_{kind}": LA.mean() / LB.mean(), f"dm_{kind}": st, f"p_{kind}": p})
                        twin_rows.append(rec)
                    rec = dict(freq=f, setting=tag, inputs=gs, pair=f"{q}/{r}")
                    for kind, sc in (("raw", None), ("norm", scale)):
                        rec.update({f"{k}_{kind}": v for k, v in equivalence(x, a, b, scale=sc).items()})
                    eq_rows.append(rec)
        print(f, flush=True)
    # nonlinear fusion rows of Fig. 2 (pooled 4h, seeds 0-2)
    scale = har_scale(setting("4h"))
    fnl, pool = run("final_fusion_nl_pool_4h"), run("final_pool_4h")
    pool = pool[(pool.task == "range") & (pool.seed <= 2) & (pool.groupset == "PMSE")]
    x = pd.concat([pool, fnl[fnl.task == "range"]])
    for enc in ("hq", "hl"):
        for fu in ("phmn", "densen"):
            rec = dict(freq="4h", setting="pooled fusion", inputs="PMSE", pair=f"{enc}_{fu}/{enc}_metan")
            for kind, sc in (("raw", None), ("norm", scale)):
                rec.update({f"{k}_{kind}": v for k, v in equivalence(
                    x, ("PMSE", f"{enc}_{fu}-pool"), ("PMSE", f"{enc}_metan-pool"), scale=sc).items()})
            eq_rows.append(rec)
    t = pd.DataFrame(twin_rows)
    for kind in ("raw", "norm"):                     # family: all rows of Table 8, as in the paper
        t[f"p_holm_{kind}"] = holm(t[f"p_{kind}"])
        t[f"star_{kind}"] = t[f"p_holm_{kind}"] < 0.05
        t[f"sig5_{kind}"] = t[f"p_{kind}"] < 0.05
    t.to_csv(OUT_T / "r5_norm_twins.csv", index=False)
    e = pd.DataFrame(eq_rows)
    e.to_csv(OUT_T / "r5_norm_equiv.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    print("Table 8: Holm stars changed:", int((t.star_raw != t.star_norm).sum()), "of", len(t),
          "| unadjusted p<.05 changed:", int((t.sig5_raw != t.sig5_norm).sum()),
          "| DM sign changed:", int((np.sign(t.dm_raw) != np.sign(t.dm_norm)).sum()))
    print(t[(t.star_raw != t.star_norm) | (t.sig5_raw != t.sig5_norm)].round(4).to_string(index=False))
    print(e.round(4).to_string(index=False))
    print("Equivalence changed:", int((e.equivalent_raw != e.equivalent_norm).sum()), "of", len(e))
    print(e[e.equivalent_raw != e.equivalent_norm].round(4).to_string(index=False))


INSANE = [("PM", "harx"), ("PMSE", "harx"), ("PMSE", "harxs"), ("PMSE", "harx-pool"), ("PMSE", "harxs-pool")]


def train_target_bounds(name):
    """(dataset, fold) -> (min, max) of the target over that fold's training rows, rebuilt from
    the run's own config (sample start, folds)."""
    import json
    from src.paper.data import make_dataset, walk_forward_folds
    cfg = json.load(open(os.path.join(run_dir(name), "config.json")))["config"]
    fc, out = cfg["folds"], {}
    for ds in cfg["datasets"]:
        coin, freq = ds.split("_")
        d = make_dataset(coin, freq, start=cfg.get("start"))
        y = d.frame["y_range"].to_numpy(float)
        for fo in walk_forward_folds(d.frame.index, fc["n"], fc["test_days"], fc["val_days"], end=fc.get("end")):
            out[(ds, fo["fold"])] = (y[fo["train"]].min(), y[fo["train"]].max())
    return out


def main_insanity():
    """Round 5: 'insanity filter' for the feature-augmented HAR models that are not ranked under
    the variance-scale losses. Every out-of-sample log forecast (validation and test, all models
    incl. HAR) is clipped to the min / max of the training-fold target of its coin before the
    validation calibration and exponentiation; then MSE (log range), QL_R, MSE_RV, QL_RV relative
    to HAR as in Table 9. Writes r5_insanity.csv and r5_insanity.tex."""
    from paper_tables_final import label
    keep = {("P", "har")} | set(INSANE)
    rows = []
    for f in ("4h", "1h", "1d"):
        E = ensemble(load_both([f"final_single_{f}", f"final_harx_{f}", f"final_har_season_{f}"]))
        E = E[[(g, m) in keep for g, m in zip(E.groupset, E.model)]].reset_index(drop=True)
        bounds = train_target_bounds(f"final_harx_{f}")
        lo = np.array([bounds[(d, k)][0] for d, k in zip(E.dataset, E.fold)])
        hi = np.array([bounds[(d, k)][1] for d, k in zip(E.dataset, E.fold)])
        C = E.assign(pred=np.clip(E.pred, lo, hi))
        clipped = (E.pred != C.pred) & (E.split == "test")
        n_clip = clipped.groupby([E.groupset, E.model]).sum()
        n_clip_val = ((E.pred != C.pred) & (E.split == "val")).groupby([E.groupset, E.model]).sum()
        n_test = (E.split == "test").groupby([E.groupset, E.model]).sum()
        for kind, X in (("raw", E), ("clipped", C)):
            T = X[X.split == "test"].assign(se=lambda d: (d.pred - d.y) ** 2)
            V = calibrated_variance(X)
            V["ql"] = qlike_loss(V.s2, V.h)
            R = rv_variance(f, X)
            for gs, m in INSANE:
                rec = dict(freq=f, filter=kind, groupset=gs, model=m, label=label(gs, m),
                           n_clipped=int(n_clip[(gs, m)]), n_test=int(n_test[(gs, m)]),
                           n_clipped_val=int(n_clip_val[(gs, m)]),
                           har_clipped=int(n_clip[("P", "har")]))
                for col, src, name in (("se", T, "mse"), ("ql", V, "ql_range"), ("mselog", R, "rv_mselog"),
                                       ("ql", R, "rv_ql")):
                    r = pair_ratio(src, (gs, m), ("P", "har"), col)
                    rec[name], rec[name + "_better"] = r["ratio"], r["better"]
                rows.append(rec)
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "r5_insanity.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    cell = lambda v, b: "--" if pd.isna(v) else (f"{v:.3f} ({b})" if v < 100 else f"$>$100 ({b})")
    with open(OUT_T / "r5_insanity.tex", "w") as fh:
        fh.write("% generated by experiments/paper_review_checks.py insanity\n")
        fh.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        fh.write("Freq. & Model, inputs & Clipped (test / val) & MSE & QL$_R$ & MSE$_{RV}$ & QL$_{RV}$ \\\\\n\\midrule\n")
        names = {"4h": "4-hourly", "1h": "Hourly", "1d": "Daily"}
        for i, f in enumerate(("4h", "1h", "1d")):
            if i:
                fh.write("\\midrule\n")
            sub = t[(t.freq == f) & (t["filter"] == "clipped")]
            for j, r in enumerate(sub.itertuples()):
                fh.write(f"{names[f] if j == 0 else ''} & {r.label} & {r.n_clipped} / {r.n_clipped_val} & {cell(r.mse, r.mse_better)} & "
                         f"{cell(r.ql_range, r.ql_range_better)} & {cell(r.rv_mselog, r.rv_mselog_better)} & "
                         f"{cell(r.rv_ql, r.rv_ql_better)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def main_lgbm_seeds():
    """Round 5: seed parity of the pooled LightGBM-S (PM): seed 0 (final_lgbm_{freq}) plus seeds 1-2
    (final_lgbm_seeds_{freq}; the seed drives the 0.8 row/column subsampling). MSE/HAR of the single
    fit, of each seed and of the 3-seed average, and the comparison with the best calendar network
    (pooled H-LSTM with group attention, PMC, three seeds). Writes r5_lgbm_seeds.csv."""
    from paper_tables_final import _pair, _vs_har, run, setting
    rows = []
    for f in ("4h", "1h", "1d"):
        df = setting(f)
        har = df[(df.groupset == "P") & (df.model == "har")]
        l1 = df[(df.groupset == "PM") & (df.model == "lgbms-pool")]
        extra = run(f"final_lgbm_seeds_{f}")
        l3 = pd.concat([l1, extra[extra.task == "range"]])
        cal = run(f"final_calendar_pool_{f}")
        net = cal[(cal.task == "range") & (cal.groupset == "PMC") & (cal.model == "hl_gattn-pool")]
        per_seed = [_vs_har(l3[l3.seed == k], har).mse.iloc[0] for k in sorted(l3.seed.unique())]
        for tag, L in (("single fit (seed 0)", l1), ("3-seed average", l3)):
            v = _vs_har(L, har).iloc[0]
            g = _pair(pd.concat([net, L]), ("PMC", "hl_gattn-pool"), ("PM", "lgbms-pool"))
            rows.append(dict(freq=f, lgbm=tag, n_seeds=L.seed.nunique(), mse_har=v.mse, better_har=int(v.better),
                             seed_min=min(per_seed), seed_max=max(per_seed),
                             net_over_lgbm=g["ratio"], net_better=g["better"], dm=g["dm"], p=g["p"]))
        print(f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "r5_lgbm_seeds.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))


if __name__ == "__main__":
    {"qlike": main_qlike, "select": main_select, "seeds": main_seeds, "altdata": main_altdata, "econ": main_econ, "pairs": main_pairs, "calendar": main_calendar, "rv": main_rv, "folds": main_folds, "harrv": main_harrv, "harmonth": main_harmonth, "equiv": main_equiv, "controls": main_controls, "matched": main_matched, "econci": main_econci, "econci_tex": main_econci_tex, "mcs_calendar": main_mcs_calendar, "loss": main_loss, "tables_holm": main_tables_holm,
     "norm_tests": main_norm_tests, "insanity": main_insanity, "lgbm_seeds": main_lgbm_seeds}[sys.argv[1]]()
