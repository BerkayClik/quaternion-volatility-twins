"""
Statistics of manuscript v5 (docs/paper/manuscript_v5): the hybrid of the hierarchical quaternion LSTM and LightGBM
as the proposed system. Everything is computed from the saved forecasts; no model is trained and no forecast changes.

Plan (2026-11-27, written and committed before any number below is computed).

  framing    Decision of 2026-11-27, taken with the advisor: the proposed system is the plain average of the H-QLSTM
             (`hq_gattn`) and the lagged LightGBM-S. This is the system of manuscript v3 and the version with which
             the system was developed. Its real twin of equal size (H-LSTM, `hl_gattn`) stays a single model of the
             tables; the hybrid built with the twin is reported in one table, the test of the algebra, and in one
             sentence of the abstract.
  known      Every table of manuscripts v3 and v4 was known when this was decided. In particular: on the development
             data the hybrid with the real twin has an MSE of 0.1825 +- 0.0002 against 0.1828 +- 0.0003 (0.14%,
             p = 0.17 over runs and 0.06 over time); it is 0.2% better on hourly bars in both tests and 0.2-0.3%
             better on the three held-out data sets in the test over time only; the two give the same range limits.
             The quaternion hybrid is better than LightGBM and than its own network by the rule on the development
             data and on the three held-out data sets (tables/v4_tests.csv, v4_newcoins.csv, v4_early_*.csv, rows
             with ours = H-QLSTM + LightGBM).
  plans      The plans of the analyses of v4 (paper_v4_paper, paper_v4_bench, paper_v4_newcoins, paper_v4_round2 to
             round4) called the hybrid with real layers "the hybrid" and named it in their criteria; for the held-out
             data the quaternion hybrid was a secondary comparison of those plans. The manuscript says so. The
             choice of the version that is presented as the system is not a result of a test: it was made with all
             results known, and the text says that too.
  what       Every analysis of v4 that was computed against the hybrid with real layers only is computed once more
             with the quaternion hybrid in its place, by one routine that has the version as a parameter. `check`
             runs the routine with the real network and requires the numbers of the v4 tables (tables/v4_*.csv) to
             the sixth digit; only then are the tables of the quaternion version (tables/v5_*.csv) used. Tests and
             rule are unchanged (paper_v3_tests: Welch over the five runs, one-sample where one side has a single
             value; Diebold-Mariano over time on the loss difference per bar averaged over runs and coins; better
             or worse only if both p < 0.05). Analyses: main (extended main table with the model confidence set),
             averages, regimes, encompass, rv, views and gains (two-by-two), extra (cost, variance scale, limits at
             three levels, pooled linear models), sets (averages of equal and higher cost, overlap, bandwidth and the
             bar of 10 October 2025 on the four data sets), flat (one encoder for all features), tuned, partners,
             time, inputs, weights (three-model schemes), example (limits of BTC for a figure). New here: twin, the
             quaternion network against its real twin and the two hybrids against each other on six data sets
             (development data on hourly, four-hour and daily bars; the three held-out data sets), MSE with the two
             tests.
  reading    Fixed here, before the numbers. Every statement that manuscript v4 makes about its hybrid is kept,
             weakened or dropped for the quaternion hybrid by the same two-test rule applied to the new numbers;
             no analysis is added that v4 did not hold, except `twin`.
             (1) "better than each single model": only for the models against which the quaternion hybrid is better
                 by the rule; exceptions are named.
             (2) held-out data: "confirmed" on a data set if the quaternion hybrid has a lower MSE than LightGBM and
                 than the H-QLSTM by the rule; the text states that the plan named the real version in this place.
             (3) tuning: "robust" if the hybrid of the tuned H-QLSTM and the tuned LightGBM is better than every
                 tuned single model by the rule, the tuned real twin included; otherwise the exception is stated.
             (4) two-by-two: the reading rule of round 3 with the H-QLSTM as the network on the window.
             (5) twin: the text says for each data set which of the two versions is better by the rule, if any. The
                 sentence "the gain is not owed to the quaternion algebra" stands if the hybrid with the twin is not
                 worse than the quaternion hybrid by the rule on any data set.
             (6) the comparison with two runs of the network and with further members is reported as the numbers
                 show it, also where it is less favourable than in v4.

    .venv/bin/python -W ignore experiments/paper_v5_stats.py check          the routine with the real network against tables/v4_*.csv
    .venv/bin/python -W ignore experiments/paper_v5_stats.py all            tables/v5_*.csv for the quaternion version
    .venv/bin/python -W ignore experiments/paper_v5_stats.py <analysis> ... single analyses
"""

import importlib
import itertools
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_paper as P3  # noqa: E402
import paper_v3_runs as RU  # noqa: E402
import paper_v4_paper as V4  # noqa: E402
import paper_v4_round2 as R2  # noqa: E402
import paper_v4_round4 as R4  # noqa: E402
from paper_v3_paper import LGBM, compare  # noqa: E402

CSV, RUNS, TRF = RU.OUT, RU.RUNS, "Transformer"
QV = SimpleNamespace(tag="q", net=P3.NET, hyb=P3.HYBRID, other_net=P3.TWIN, other_hyb=P3.TWIN_HYBRID, lab="H-QLSTM", other_lab="H-LSTM", key="hq_gattn-pool",
                     flat="qlstm-pool", flat_lab="Single QLSTM", other_word="real")
RV = SimpleNamespace(tag="r", net=P3.TWIN, hyb=P3.TWIN_HYBRID, other_net=P3.NET, other_hyb=P3.HYBRID, lab="H-LSTM", other_lab="H-QLSTM", key="hl_gattn-pool",
                     flat="lstm-pool", flat_lab="Single LSTM", other_word="quaternion")
SETS = ("design", "newcoins", "early_study", "early_new")
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
mix = lambda *fs: fs[0].assign(pred=sum(f.reindex(fs[0].index).pred for f in fs) / len(fs))
pd.set_option("display.width", 280, "display.max_columns", 40, "display.max_rows", 200)

_CACHE: dict = {}


def M0() -> dict:
    if "M" not in _CACHE:
        _CACHE["M"] = P3.build()[0]
    return _CACHE["M"]


def caps() -> pd.Series:
    if "caps" not in _CACHE:
        import paper_v2_round2 as R
        _CACHE["caps"] = R.train_caps()
    return _CACHE["caps"]


def _ref():
    """Index and targets of the development data, and helpers to put forecasts on it."""
    f = M0()[P3.TWIN][0]
    ref, y = f.index, f.y.to_numpy()
    frame = lambda pred: pd.DataFrame({"y": y, "pred": np.asarray(pred)}, index=ref)
    pred = lambda g: g.reindex(ref).pred.to_numpy()
    return ref, y, frame, pred


# ============================================================================================ main table
def main(v) -> dict:
    """paper_v4_round2.main_ext with the version as a parameter."""
    from arch.bootstrap import MCS
    M, get, L = M0(), RU.forecasts, v.lab
    ref, y, frame, pred = _ref()
    X = {n: M[n] for n in R2.MAIN}
    X["GARCH(1,1)"] = {0: get(("P", "garch"), ["results/v4bench/garch"])[0].reindex(ref)}
    X["GJR-GARCH"] = {0: get(("P", "gjr"), ["results/v4bench/garch"])[0].reindex(ref)}
    X["HGB"] = {k: f.reindex(ref) for k, f in get(("PM", "hgbsl-pool"), ["results/v4bench/hgb"]).items()}
    X["CARR"] = {0: get(("P", "carr"), ["results/v4views/linear"])[0].reindex(ref)}
    X["Linear, pooled"] = {0: get(("PM", "linear-pool"), ["results/v4views/linear"])[0].reindex(ref)}
    avg = lambda *names: {k: frame(np.mean([pred(X[n][k]) for n in names], axis=0)) for k in RUNS}
    mean5 = np.mean([pred(M[v.net][k]) for k in RUNS], axis=0)
    X[f"Five runs of {L}"] = {0: frame(mean5)}
    X["LightGBM + HGB"], X[f"{L} + HGB"] = avg(LGBM, "HGB"), avg(v.net, "HGB")
    X["Transformer + LightGBM"], X[f"{L} + Transformer"] = avg(TRF, LGBM), avg(v.net, TRF)
    X[f"{L} + LightGBM + Transformer"] = avg(v.net, LGBM, TRF)
    X[f"Five runs of {L} + LightGBM"] = {k: frame((mean5 + pred(M[LGBM][k])) / 2) for k in RUNS}
    per_run, per_bar = V4._losses(X, caps())
    har = per_run["HAR"].mean()
    rows = []
    for name in X:
        p = per_run[name]
        r = dict(ours=v.hyb, model=name, runs=len(p), mse_mean=p.mse.mean(), mse_sd=sd(p.mse), mse_har=p.mse.mean() / har.mse, qlike_mean=p.qlike.mean(),
                 qlike_sd=sd(p.qlike), qlike_median=per_bar[name].qlike.median())
        if name != v.hyb:
            r.update(compare(per_run, per_bar, v.hyb, name))
        rows.append(r)
    t = V4._holm(pd.DataFrame(rows), ["ours"], ["mse_p_runs", "mse_p_time", "qlike_p_runs", "qlike_p_time"])
    Lm = pd.DataFrame({n: per_bar[n].mse.unstack("dataset").mean(axis=1) for n in X}).dropna()
    m = MCS(Lm.to_numpy(), size=0.10, reps=2000, block_size=14, seed=0)
    m.compute()
    pv = m.pvalues.squeeze()
    t["mcs_p"] = [float(pv.loc[i]) for i in range(len(Lm.columns))]
    return {"main": (t, ["model"])}


# ============================================================================================ averages
def averages(v) -> dict:
    """paper_v4_round2.averages: three-member average, members chosen on the validation block, averages of five runs."""
    M, L, S = M0(), v.lab, R2.SINGLES
    ref, y, frame, _ = _ref()
    one = lambda name, k: M[name][k if len(M[name]) > 1 else 0]
    fold = ref.get_level_values("fold").to_numpy()
    is_val = ref.get_level_values("split").to_numpy() == "val"
    P = {k: pd.DataFrame({n: one(n, k).reindex(ref).pred.to_numpy() for n in S}, index=ref) for k in RUNS}
    assert all(not p.isna().any().any() for p in P.values())
    new, chosen = {}, []
    named = {f"{L} + LightGBM + Transformer": [v.net, LGBM, TRF], "H-LSTM + H-QLSTM + LightGBM": [P3.TWIN, P3.NET, LGBM],
             "H-LSTM + H-QLSTM + LightGBM + Transformer": [P3.TWIN, P3.NET, LGBM, TRF], "All eleven single models": S}
    for name, members in named.items():
        new[("named", name)] = {k: frame(P[k][members].mean(axis=1)) for k in RUNS}
    val_mse = lambda p_, f: float(np.mean((p_[is_val & (fold == f)] - y[is_val & (fold == f)]) ** 2))
    for kk in (1, 2, 3, 4, 5):
        out = {}
        for k in RUNS:
            p_ = np.full(len(ref), np.nan)
            for f in np.unique(fold):
                rank = sorted(S, key=lambda n: val_mse(P[k][n].to_numpy(), f))[:kk]
                p_[fold == f] = P[k].loc[fold == f, rank].mean(axis=1)
                chosen.append(dict(scheme=f"top-{kk}", run=k, fold=int(f), members=" | ".join(sorted(rank))))
            out[k] = frame(p_)
        new[("top-k", f"Best {kk} on the validation block")] = out
    out = {}
    for k in RUNS:
        p_ = np.full(len(ref), np.nan)
        for f in np.unique(fold):
            a, b = min(itertools.combinations(S, 2), key=lambda ab: val_mse(P[k][list(ab)].mean(axis=1).to_numpy(), f))
            p_[fold == f] = P[k].loc[fold == f, [a, b]].mean(axis=1)
            chosen.append(dict(scheme="pair", run=k, fold=int(f), members=" | ".join(sorted([a, b]))))
        out[k] = frame(p_)
    new[("pair", "Best pair on the validation block")] = out
    mean5 = {n: np.mean([P[k][n].to_numpy() for k in RUNS], axis=0) for n in (v.net, LGBM, TRF)}
    for n, label in ((v.net, L), (LGBM, "LightGBM"), (TRF, "Transformer")):
        new[("cost", f"Five runs of {label}")] = {0: frame(mean5[n])}
    new[("cost", f"Five runs of {L} + five runs of LightGBM")] = {0: frame((mean5[v.net] + mean5[LGBM]) / 2)}
    allm = {v.hyb: M[v.hyb], LGBM: M[LGBM], **{name: f for (_, name), f in new.items()}}
    per_run, per_bar = V4._losses(allm, caps())
    rows = []
    for (kind, name) in new:
        p = per_run[name]
        rows.append(dict(kind=kind, name=name, runs=len(p), mse_mean=p.mse.mean(), mse_sd=sd(p.mse), qlike_mean=p.qlike.mean(), qlike_sd=sd(p.qlike),
                         **{"hyb_" + c: x for c, x in compare(per_run, per_bar, name, v.hyb).items()},
                         **{"lgbm_" + c: x for c, x in compare(per_run, per_bar, name, LGBM).items() if c.startswith("mse_")}))
    return {"averages": (pd.DataFrame(rows), ["name"]), "selected": (pd.DataFrame(chosen), None)}


# ============================================================================================ regimes, encompassing, realised variance
def regimes(v) -> dict:
    M = M0()
    names = ("HAR", "HAR-S x", LGBM, v.net, v.other_hyb, v.hyb)
    Lm = R2._limits(M, list(names))
    har = M["HAR"][0].reset_index()
    q = har[har.split == "val"].groupby(["dataset", "fold"]).pred.quantile([1 / 3, 2 / 3]).unstack().rename(columns={1 / 3: "q1", 2 / 3: "q2"})
    ht = har[har.split == "test"].join(q, on=["dataset", "fold"])
    ht["regime"] = np.where(ht.pred <= ht.q1, "low", np.where(ht.pred <= ht.q2, "middle", "high"))
    regime = ht.set_index(["dataset", "time"]).regime
    in_regime = lambda test, reg: test[test.set_index(["dataset", "time"]).index.map(regime).to_numpy() == reg]
    rows = []
    for reg in ("low", "middle", "high"):
        stat = {}
        for name in names:
            parts = [in_regime(test, reg) for test in Lm[name]]
            bars = [x.set_index(["time", "dataset"]).tick.sort_index() for x in parts]
            stat[name] = dict(runs=np.array([x.tick.mean() for x in parts]), bar=sum(bars) / len(bars), n=len(parts[0]),
                              exceed=float(np.mean([x.hit.mean() for x in parts])))
        ours = stat[v.hyb]
        for name, s_ in stat.items():
            r = dict(regime=reg, bars=s_["n"], model=name, exceed=s_["exceed"], tick_mean=s_["runs"].mean(), tick_sd=sd(s_["runs"]))
            if name != v.hyb:
                r.update(R2._two_tests(ours["runs"], s_["runs"], ours["bar"], s_["bar"]))
            rows.append(r)
    return {"regimes": (pd.DataFrame(rows), ["regime", "model"])}


def encompass(v) -> dict:
    from src.paper.evaluate import diebold_mariano
    M = M0()
    test = lambda f: f[f.index.get_level_values("split") == "test"]
    err = {n: {k: (lambda f: (f.y - f.pred))(test(M[n][k if len(M[n]) > 1 else 0])) for k in RUNS} for n in (LGBM, v.net, TRF, "HAR-S x", v.hyb)}
    pairs = [(LGBM, v.net), (v.net, LGBM), (v.hyb, LGBM), (v.hyb, v.net), (v.hyb, TRF), (v.hyb, "HAR-S x")]
    rows = []
    for a, b in pairs:
        d = sum(err[a][k] * (err[a][k] - err[b][k].reindex(err[a][k].index)) for k in RUNS) / len(RUNS)
        lam = float(np.mean([(err[a][k] * (err[a][k] - err[b][k].reindex(err[a][k].index))).sum() / ((err[a][k] - err[b][k].reindex(err[a][k].index)) ** 2).sum() for k in RUNS]))
        series = d.droplevel(["fold", "split"]).unstack("dataset").mean(axis=1).sort_index().to_numpy()
        stat, p = diebold_mariano(series, np.zeros(len(series)))
        rows.append(dict(model=a, other=b, weight_on_other=lam, mean=float(series.mean()), statistic=float(stat), p=float(p),
                         encompasses=bool(p >= 0.05 or series.mean() < 0)))
    return {"encompass": (pd.DataFrame(rows), ["model", "other"])}


def rv(v) -> dict:
    M = M0()
    names = ["HAR", "HAR-S x", LGBM, TRF, P3.NET, P3.TWIN, P3.HYBRID, P3.TWIN_HYBRID]
    step = pd.Timedelta(hours=4)
    real = {}
    for ds in M["HAR"][0].index.get_level_values("dataset").unique():
        r = pd.read_csv(f"data/binance_1m/{ds.split('_')[0]}_4h_rv.csv", index_col=0, parse_dates=True)
        r = r[r.n_min >= 0.9 * 240].rv
        r.index = r.index.tz_localize(None) if r.index.tz is None else r.index.tz_convert(None)
        real[ds] = r
    per_run, per_bar, n_bars = {}, {}, None
    for name in names:
        runs, bars = [], []
        for f in M[name].values():
            d = f.reset_index()
            t = pd.to_datetime(d.time)
            t = t.dt.tz_convert(None) if t.dt.tz is not None else t
            d["rv"] = np.nan
            for ds, g in d.groupby("dataset"):
                d.loc[g.index, "rv"] = real[ds].reindex(t[g.index] + step).to_numpy()
            d = d[np.isfinite(d.rv) & (d.rv > 0)]
            d["raw"] = np.exp(2 * d.pred)
            vv = d[d.split == "val"].groupby(["dataset", "fold"]).agg(rv=("rv", "mean"), raw=("raw", "mean"))
            te = d[d.split == "test"].join((vv.rv / vv.raw).rename("c"), on=["dataset", "fold"])
            te = te[np.isfinite(te.c)]
            h = te.c * te.raw
            te = te.assign(qlike=te.rv / h - np.log(te.rv / h) - 1, mselog=(np.log(te.rv) - np.log(h)) ** 2)
            runs.append(te[["qlike", "mselog"]].mean())
            bars.append(te.set_index(["time", "dataset"])[["qlike", "mselog"]].sort_index())
        per_run[name], per_bar[name] = pd.DataFrame(runs), sum(bars) / len(bars)
        n_bars = len(bars[0])
    rows = []
    for name in names:
        r = dict(model=name, runs=len(per_run[name]), test_coin_bars=n_bars)
        for loss in ("mselog", "qlike"):
            x = per_run[name][loss].to_numpy()
            r.update({f"{loss}_mean": x.mean(), f"{loss}_sd": sd(x), f"{loss}_har": x.mean() / per_run["HAR"][loss].mean()})
            if name != v.hyb:
                r.update({f"{loss}_{c}": z for c, z in R2._two_tests(per_run[v.hyb][loss].to_numpy(), x, per_bar[v.hyb][loss], per_bar[name][loss]).items()})
        rows.append(r)
    return {"rv": (pd.DataFrame(rows), ["model"])}


# ============================================================================================ two-by-two
def _cells(v) -> dict:
    import paper_v4_round3 as R3
    M = M0()
    return {f"network, window ({v.lab})": M[v.net], "tree, summaries (LightGBM)": M[LGBM],
            "tree, window": RU.forecasts(("PMW", "lgbmw-pool"), [R3.VIEWS_DIR]), "network, summaries": RU.forecasts(("PRMC", "smlp-pool"), [R3.SMLP_RUNS])}


def views(v) -> dict:
    """paper_v4_round3.views and paper_v4_round4.gains with the network of the version on the window."""
    from scipy import stats
    M = M0()
    ref, y, frame, _ = _ref()
    test = ref.get_level_values("split").to_numpy() == "test"
    cells = _cells(v)
    P = {n: {k: r[k].reindex(ref).pred.to_numpy() for k in RUNS} for n, r in cells.items()}
    assert all(np.isfinite(x).all() for r in P.values() for x in r.values())
    learner, view = (lambda n: n.split(",")[0]), (lambda n: n.split(",")[1].split()[0])
    kind = lambda a, b: "view only" if learner(a) == learner(b) else "learner only" if view(a) == view(b) else "both"
    allm = {n: {k: frame(P[n][k]) for k in RUNS} for n in cells}
    pairs = list(itertools.combinations(cells, 2))
    for a, b in pairs:
        allm[f"{a} + {b}"] = {k: frame((P[a][k] + P[b][k]) / 2) for k in RUNS}
    allm[v.hyb] = M[v.hyb]
    per_run, per_bar = V4._losses(allm, caps())
    rows = [dict(kind="single", name=n, mse_mean=per_run[n].mse.mean(), mse_sd=sd(per_run[n].mse), qlike_mean=per_run[n].qlike.mean()) for n in cells]
    G, grow = {}, []
    for a, b in pairs:
        name = f"{a} + {b}"
        corr = float(np.mean([np.corrcoef(P[a][k][test] - y[test], P[b][k][test] - y[test])[0, 1] for k in RUNS]))
        ma, mb, mp = per_run[a].mse.mean(), per_run[b].mse.mean(), per_run[name].mse.mean()
        r = dict(kind=kind(a, b), name=name, mse_mean=mp, mse_sd=sd(per_run[name].mse), qlike_mean=per_run[name].qlike.mean(), corr=corr,
                 gain_pct=100 * (1 - mp / ((ma + mb) / 2)), vs_better_pct=100 * (mp / min(ma, mb) - 1))
        r.update({"hyb_" + c: x for c, x in compare(per_run, per_bar, name, v.hyb).items() if c.startswith("mse_")})
        rows.append(r)
        g = 100 * (1 - per_run[name].mse.to_numpy() / ((per_run[a].mse.to_numpy() + per_run[b].mse.to_numpy()) / 2))
        G[(a, b)] = g
        grow.append(dict(kind={"view only": "input only", "learner only": "kind of model only", "both": "both"}[kind(a, b)],
                         pair=f"{a.split(' (')[0]} + {b.split(' (')[0]}", gain_mean=g.mean(), gain_sd=sd(g), gain_min=g.min(), gain_max=g.max()))
    t = pd.DataFrame(rows)
    ours = f"network, window ({v.lab}) + tree, summaries (LightGBM)"
    assert np.isclose(t.loc[t.name == ours, "mse_mean"].iloc[0], per_run[v.hyb].mse.mean(), rtol=1e-9)
    m = lambda k_: np.mean([g for (a, b), g in G.items() if kind(a, b) == k_], axis=0)
    d = m("learner only") - m("view only")
    half = stats.t.ppf(0.975, len(d) - 1) * sd(d) / np.sqrt(len(d))
    grow.append(dict(kind="difference", pair="kind of model only minus input only (means of two pairs)", gain_mean=d.mean(), gain_sd=sd(d), gain_min=d.mean() - half,
                     gain_max=d.mean() + half, p=float(stats.ttest_1samp(d, 0).pvalue)))
    return {"views": (t, ["name"]), "gains": (pd.DataFrame(grow), ["pair"])}


# ============================================================================================ cost, variance scale, levels, linear
def extra(v) -> dict:
    """paper_v4_round3.extra with the version as a parameter."""
    import paper_v2_round2 as R
    import paper_v4_round3 as R3
    get, L = RU.forecasts, v.lab
    M = dict(M0())
    HYB, NET = v.hyb, v.net
    ref, y, frame, _ = _ref()
    P = {n: {k: M[n][k].reindex(ref).pred.to_numpy() for k in RUNS} for n in (NET, LGBM, TRF)}
    TRIPLE = f"{L} + LightGBM + Transformer"
    M[TRIPLE] = {k: frame((P[NET][k] + P[LGBM][k] + P[TRF][k]) / 3) for k in RUNS}
    out = {}

    mean5 = np.mean([P[NET][k] for k in RUNS], axis=0)
    X = {HYB: M[HYB], TRIPLE: M[TRIPLE],
         f"Two runs of {L} + LightGBM": {k: frame((P[NET][k] + P[NET][(k + 1) % 5] + P[LGBM][k]) / 3) for k in RUNS},
         f"Five runs of {L} + LightGBM": {k: frame((mean5 + P[LGBM][k]) / 2) for k in RUNS}}
    per_run, per_bar = V4._losses(X, caps())
    rows = [dict(name=n, mse_mean=per_run[n].mse.mean(), mse_sd=sd(per_run[n].mse), **{"hyb_" + c: x for c, x in compare(per_run, per_bar, n, HYB).items() if c.startswith("mse_")},
                 **{"triple_" + c: x for c, x in compare(per_run, per_bar, n, TRIPLE).items() if c.startswith("mse_")}) for n in list(X)[2:]]
    out["cost"] = (pd.DataFrame(rows), ["name"])

    cal = lambda f: R._calibrated(f.reset_index().assign(groupset="g", model="m"), None, caps=caps()).set_index(["time", "dataset"]).sort_index()
    runs, bars = [], []
    for k in RUNS:
        a, b = cal(M[NET][k]), cal(M[LGBM][k])
        h = (a.h + b.h.reindex(a.index)) / 2
        ql = a.s2 / h - np.log(a.s2 / h) - 1
        runs.append(float(ql.mean())); bars.append(ql)
    v_run, v_bar = np.array(runs), sum(bars) / len(bars)
    pr, pb = V4._losses({HYB: M[HYB], LGBM: M[LGBM]}, caps())
    rows = [dict(name="hybrid on the variance scale", qlike_mean=v_run.mean(), qlike_sd=sd(v_run), other=o, other_qlike=pr[o].qlike.mean(),
                 **R2._two_tests(v_run, pr[o].qlike.to_numpy(), v_bar, pb[o].qlike)) for o in (LGBM, HYB)]
    out["variance"] = (pd.DataFrame(rows), ["other"])

    names = ["HAR", "HAR-S x", LGBM, NET, TRF, HYB, TRIPLE]
    rows = []
    for level in (0.95, 0.99, 0.995):
        Lm = R2._limits(M, names, rate=1 - level)
        stat = {}
        for n in names:
            runs = Lm[n]
            excess = [np.exp((x.y - x.pred - x.c)[x.hit == 1]) - 1 for x in runs]
            bars = [x.set_index(["time", "dataset"]).tick.sort_index() for x in runs]
            stat[n] = dict(tick=np.array([x.tick.mean() for x in runs]), bar=sum(bars) / len(bars), exceed=np.mean([x.hit.mean() for x in runs]),
                           excess_mean=np.mean([e.mean() for e in excess]), excess_max=np.max([e.max() for e in excess]))
        if level in R3.QUANTILES:
            q = get(("PM", "lgbmq-pool"), [f"{R3.EXTRA_DIR}/lgbmq{int(round(100 * level))}"])
            runs = []
            for k in RUNS:
                x = q[k].reset_index()
                x = x[x.split == "test"].copy()
                u = x.y - x.pred
                x["hit"], x["tick"] = (u > 0).astype(int), u * (level - (u < 0))
                runs.append(x)
            bars = [x.set_index(["time", "dataset"]).tick.sort_index() for x in runs]
            excess = [np.exp((x.y - x.pred)[x.hit == 1]) - 1 for x in runs]
            stat["LightGBM, quantile objective"] = dict(tick=np.array([x.tick.mean() for x in runs]), bar=sum(bars) / len(bars), exceed=np.mean([x.hit.mean() for x in runs]),
                                                        excess_mean=np.mean([e.mean() for e in excess]), excess_max=np.max([e.max() for e in excess]))
        for n, s_ in stat.items():
            r = dict(level=level, model=n, exceed=s_["exceed"], tick_mean=s_["tick"].mean(), tick_sd=sd(s_["tick"]), tick_har=s_["tick"].mean() / stat["HAR"]["tick"].mean(),
                     excess_mean=s_["excess_mean"], excess_max=s_["excess_max"])
            if n != HYB:
                r.update(R2._two_tests(stat[HYB]["tick"], s_["tick"], stat[HYB]["bar"], s_["bar"]))
            rows.append(r)
    out["levels"] = (pd.DataFrame(rows), ["level", "model"])

    D = f"{R3.EXTRA_DIR}/linear"
    lin = {"HAR, pooled": {0: get(("PM", "har-pool"), [D])[0].reindex(ref)}, "HAR-S, pooled": {0: get(("PM", "hars-pool"), [D])[0].reindex(ref)},
           "Linear on the 35 regressors, pooled": {0: get(("PM", "linear-pool"), [D])[0].reindex(ref)}, "CARR": {0: get(("P", "carr"), [D])[0].reindex(ref)}}
    lin[f"{L} + linear on the 35 regressors"] = {k: frame((P[NET][k] + lin["Linear on the 35 regressors, pooled"][0].pred.to_numpy()) / 2) for k in RUNS}
    per_run, per_bar = V4._losses({HYB: M[HYB], LGBM: M[LGBM], "HAR": M["HAR"], **lin}, caps())
    rows = [dict(model=n, mse_mean=per_run[n].mse.mean(), mse_sd=sd(per_run[n].mse), mse_har=per_run[n].mse.mean() / per_run["HAR"].mse.mean(), qlike_mean=per_run[n].qlike.mean(),
                 **{"hyb_" + c: x for c, x in compare(per_run, per_bar, HYB, n).items() if c.startswith("mse_")},
                 **{"lgbm_" + c: x for c, x in compare(per_run, per_bar, LGBM, n).items() if c.startswith("mse_")}) for n in lin]
    out["linear"] = (pd.DataFrame(rows), ["model"])
    return out


# ============================================================================================ the four data sets
def _base(tag: str) -> tuple[dict, pd.Series]:
    """Forecasts of one of the four data sets (HAR, LightGBM, Transformer and both versions of the network) and its QLIKE caps."""
    if ("set", tag) in _CACHE:
        return _CACHE[("set", tag)]
    if tag == "design":
        M = M0()
        out = ({"HAR": M["HAR"], "LightGBM": M[LGBM], "H-QLSTM": M[P3.NET], "H-LSTM": M[P3.TWIN], TRF: M[TRF]}, caps())
    else:
        import paper_v4_newcoins as N
        importlib.reload(N)
        if tag.startswith("early"):
            import paper_v4_round3 as R3
            N = R3._early(tag.split("_")[1])
        fr = N.frames()
        c = pd.DataFrame([dict(dataset=ds, fold=k, cap=float(np.exp(2 * np.nanmax(d.frame["y_range"].to_numpy(float)[fo["train"]]))))
                          for ds, (d, folds) in fr.items() for k, fo in enumerate(folds)]).set_index(["dataset", "fold"]).cap
        get = RU.forecasts
        M = {"HAR": {0: get(("P", "har"), [N.BENCH])[0]}, "LightGBM": get(("PM", "lgbmsl-pool"), [N.BENCH]), "H-QLSTM": get(("PRMC", "hq_gattn-pool"), [N.OUT_RUNS]),
             "H-LSTM": get(("PRMC", "hl_gattn-pool"), [N.OUT_RUNS]), TRF: get(("PRMC", "transformer-pool"), [N.OUT_RUNS])}
        assert all(sorted(M[n]) == list(RUNS) for n in ("LightGBM", "H-QLSTM", "H-LSTM", TRF)), tag
        out = (M, c)
    _CACHE[("set", tag)] = out
    return out


def _loaded(tag: str, v) -> dict:
    """Per-bar losses of the models of a data set and of the averages built with the network of the version."""
    if ("bars", tag, v.tag) not in _CACHE:
        M, c = _base(tag)
        L, N_, G, T = v.lab, M[v.lab], M["LightGBM"], M[TRF]
        X = {"HAR": M["HAR"], "LightGBM": G, L: N_, TRF: T}
        X[f"{L} + LightGBM"] = {k: mix(N_[k], G[k]) for k in RUNS}
        X[f"{L} + LightGBM + Transformer"] = {k: mix(N_[k], G[k], T[k]) for k in RUNS}
        X[f"Two runs of {L}"] = {k: mix(N_[k], N_[(k + 1) % 5]) for k in RUNS}
        X[f"Five runs of {L}"] = {0: mix(*[N_[k] for k in RUNS])}
        X[f"Two runs of {L} + LightGBM"] = {k: mix(N_[k], N_[(k + 1) % 5], G[k]) for k in RUNS}
        X[f"Five runs of {L} + LightGBM"] = {k: mix(X[f"Five runs of {L}"][0], G[k]) for k in RUNS}
        X[f"{L} + Transformer"] = {k: mix(N_[k], T[k]) for k in RUNS}
        X["Transformer + LightGBM"] = {k: mix(T[k], G[k]) for k in RUNS}
        _CACHE[("bars", tag, v.tag)] = R4._bars(X, c)
    return _CACHE[("bars", tag, v.tag)]


def sets(v) -> dict:
    """paper_v4_round4: averages of equal and higher cost, overlap, bandwidth and the bar of 10 October 2025."""
    L = v.lab
    HYB, TRIPLE = f"{L} + LightGBM", f"{L} + LightGBM + Transformer"
    time = lambda idx: pd.DatetimeIndex(idx.get_level_values("time"))
    rows = []
    for tag in SETS:
        B = _loaded(tag, v)
        for other in (f"Two runs of {L}", f"Five runs of {L}", f"Two runs of {L} + LightGBM", f"Five runs of {L} + LightGBM", TRIPLE, f"{L} + Transformer",
                      "Transformer + LightGBM"):
            rows.append(dict(data=tag, **R4._cmp(B, HYB, other)))
        for other in (f"Two runs of {L} + LightGBM", f"Five runs of {L} + LightGBM"):
            rows.append(dict(data=tag, **R4._cmp(B, TRIPLE, other)))
    out = {"sets_averages": (pd.DataFrame(rows), ["data", "a", "b"])}

    f0 = M0()[P3.TWIN][0].reset_index()
    start = f0[(f0.split == "val") & (f0.fold == 0)].time.min()
    B = _loaded("early_study", v)
    keep = lambda idx: np.asarray(time(idx) < start)
    rows = [dict(data="early_study", cut=str(start), **R4._cmp(B, HYB, o, keep=keep)) for o in ("LightGBM", L, "HAR")]
    rows += [dict(data="early_study", cut="none", **R4._cmp(B, HYB, o)) for o in ("LightGBM", L, "HAR")]
    out["sets_overlap"] = (pd.DataFrame(rows), ["cut", "a", "b"])

    rows = [dict(data=tag, lag=lag, **R4._cmp(_loaded(tag, v), HYB, o, lag=lag)) for tag in SETS for o in ("LightGBM", L) for lag in (14, 42, 84)]
    out["sets_bandwidth"] = (pd.DataFrame(rows), ["data", "lag", "a", "b"])

    B = _loaded("design", v)
    t = time(B[HYB][0].index)
    bar = pd.Timestamp("2025-10-10 20:00", tz=t.tz)
    lo, hi = bar - pd.Timedelta(hours=4), bar + pd.Timedelta(hours=4 * 43)
    keep = lambda idx: np.asarray((time(idx) < lo) | (time(idx) > hi))
    rows = [dict(sample="without the bar and its windows", **R4._cmp(B, HYB, o, keep=keep)) for o in ("HAR", "LightGBM", L)]
    rows += [dict(sample="all bars", **R4._cmp(B, HYB, o)) for o in ("HAR", "LightGBM", L)]
    out["sets_print"] = (pd.DataFrame(rows), ["sample", "a", "b"])
    return out


def twin(v=None) -> dict:
    """The quaternion network against its real twin, and the hybrid of each with LightGBM against the other, on six
    data sets: MSE, a (quaternion) against b (real), with the two tests. New in v5; no counterpart in v4."""
    import paper_v2_round2 as R
    import paper_v3_bars as B3
    rows = []
    for tag in SETS:
        M, c = _base(tag)
        X = {"H-QLSTM": M["H-QLSTM"], "H-LSTM": M["H-LSTM"], "H-QLSTM + LightGBM": {k: mix(M["H-QLSTM"][k], M["LightGBM"][k]) for k in RUNS},
             "H-LSTM + LightGBM": {k: mix(M["H-LSTM"][k], M["LightGBM"][k]) for k in RUNS}}
        B = R4._bars(X, c)
        for what, a, b in (("network", "H-QLSTM", "H-LSTM"), ("hybrid", "H-QLSTM + LightGBM", "H-LSTM + LightGBM")):
            rows.append(dict(data=tag, freq="4h", what=what, **R4._cmp(B, a, b)))
    for freq in ("1h", "1d"):
        c = R.train_caps(freq)
        runs = {name: RU.forecasts(key, pats) for name, (key, pats) in B3.SOURCES[freq].items()}
        X = {"H-QLSTM": {k: runs[P3.NET][k] for k in RUNS}, "H-LSTM": {k: runs[P3.TWIN][k] for k in RUNS}}
        X["H-QLSTM + LightGBM"] = {k: mix(runs[P3.NET][k], runs[LGBM][k]) for k in RUNS}
        X["H-LSTM + LightGBM"] = {k: mix(runs[P3.TWIN][k], runs[LGBM][k]) for k in RUNS}
        B = {n: [B3.bar_losses(f, c) for f in r.values()] for n, r in X.items()}
        for what, a, b in (("network", "H-QLSTM", "H-LSTM"), ("hybrid", "H-QLSTM + LightGBM", "H-LSTM + LightGBM")):
            rows.append(dict(data="design", freq=freq, what=what, **R4._cmp(B, a, b)))
    return {"twin": (pd.DataFrame(rows), ["data", "freq", "what"])}


# ============================================================================================ one encoder for all features
def flat(v) -> dict:
    import paper_v3_tables as T
    pattern = [g[2] for g in T.GRID if g[1] == "flat QLSTM / LSTM"][0]
    F = RU.forecasts(("PRMC", v.flat), [pattern])
    runs, M, L, S = sorted(F), M0(), v.lab, v.flat_lab
    ref = M[P3.TWIN][0].index
    X = {L: {k: M[v.net][k] for k in runs}, "LightGBM": {k: M[LGBM][k] for k in runs}, f"{L} + LightGBM": {k: M[v.hyb][k] for k in runs},
         S: {k: F[k].reindex(ref) for k in runs}}
    assert all(np.isfinite(f.pred).all() for f in X[S].values())
    X[f"{S} + LightGBM"] = {k: mix(X[S][k], X["LightGBM"][k]) for k in runs}
    B = R4._bars(X, caps())
    rows = [dict(runs=len(runs), **R4._cmp(B, a, b)) for a, b in ((S, L), (f"{S} + LightGBM", f"{L} + LightGBM"), (f"{S} + LightGBM", "LightGBM"), (f"{S} + LightGBM", S))]
    return {"flat": (pd.DataFrame(rows), ["a", "b"])}


# ============================================================================================ tuned benchmarks
def tuned(v) -> dict:
    """The tuned part of paper_v4_bench.evaluate: the hybrid of the tuned network of the version and the tuned LightGBM."""
    import paper_v4_bench as BE
    get, M, L = RU.forecasts, M0(), v.lab
    ref, y, frame, _ = _ref()
    fold = ref.get_level_values("fold").to_numpy()
    is_val = ref.get_level_values("split").to_numpy() == "val"
    mixp = lambda a, b: frame((a.pred.to_numpy() + b.pred.to_numpy()) / 2)
    val_mse = lambda p_, f: float(np.mean((p_[is_val & (fold == f)] - y[is_val & (fold == f)]) ** 2))
    tun, default, choices = {}, {}, []
    for label, settings in BE._families().items():
        runs = {}
        for s_, (key, pats) in settings.items():
            r = get(key, pats)
            if not set(RUNS) <= set(r):
                print(f"incomplete: {label}, {s_}: runs {sorted(r)}")
                continue
            runs[s_] = {k: r[k].reindex(ref).pred.to_numpy() for k in RUNS}
            assert all(np.isfinite(x).all() for x in runs[s_].values()), (label, s_)
        first = next(iter(settings))
        default[label] = {k: frame(runs[first][k]) for k in RUNS}
        tun[label] = {}
        for k in RUNS:
            p_ = np.full(len(ref), np.nan)
            for f in np.unique(fold):
                best = min(runs, key=lambda s_: val_mse(runs[s_][k], f))
                p_[fold == f] = runs[best][k][fold == f]
                choices.append(dict(family=label, run=k, fold=int(f), setting=best))
            tun[label][k] = frame(p_)
    fixed = {n: {0: M[n][0].reindex(ref)} for n in ("HAR", "HAR-S x")}
    other = f"Hybrid, {v.other_word} (tuned)"
    allm = {**{f"{n} (default)": x for n, x in default.items()}, **{f"{n} (tuned)": x for n, x in tun.items()}, **fixed,
            "Hybrid (default)": M[v.hyb], "Hybrid (tuned)": {k: mixp(tun[L][k], tun["LightGBM"][k]) for k in RUNS},
            other: {k: mixp(tun[v.other_lab][k], tun["LightGBM"][k]) for k in RUNS}}
    per_run, per_bar = V4._losses(allm, caps())
    rows = []
    for label in list(tun) + ["HAR", "HAR-S x"]:
        t_name, d_name = (f"{label} (tuned)", f"{label} (default)") if label in tun else (label, label)
        pt, pdf = per_run[t_name], per_run[d_name]
        r = dict(family=label, default_mse=pdf.mse.mean(), default_sd=sd(pdf.mse), tuned_mse=pt.mse.mean(), tuned_sd=sd(pt.mse), tuned_qlike=pt.qlike.mean())
        if label in tun:
            r.update({"gain_" + c: x for c, x in compare(per_run, per_bar, t_name, d_name).items() if c.startswith("mse_")})
        r.update({"tunedhyb_" + c: x for c, x in compare(per_run, per_bar, "Hybrid (tuned)", t_name).items()})
        r.update({"defaulthyb_" + c: x for c, x in compare(per_run, per_bar, "Hybrid (default)", t_name).items() if c.startswith("mse_")})
        rows.append(r)
    for name in ("Hybrid (default)", "Hybrid (tuned)", other):
        p_ = per_run[name]
        r = dict(family=name, tuned_mse=p_.mse.mean(), tuned_sd=sd(p_.mse), tuned_qlike=p_.qlike.mean())
        if name != "Hybrid (tuned)":
            r.update({"tunedhyb_" + c: x for c, x in compare(per_run, per_bar, "Hybrid (tuned)", name).items()})
        rows.append(r)
    return {"tuned": (pd.DataFrame(rows), ["family"])}


# ============================================================================================ partners, time, inputs, weights, example
def partners(v) -> dict:
    """Plain averages of two single models against the hybrid of the version (the partners and the time block of paper_v4_paper.stats)."""
    M = dict(M0())
    net, oth = v.net, v.other_net
    pairs = [(net, LGBM), (oth, LGBM), (TRF, LGBM), ("PatchTST", LGBM), ("iTransformer", LGBM), ("HAR-S x", LGBM), (LGBM, LGBM),
             (net, net), (oth, oth), (net, oth), (net, TRF), (net, "PatchTST"), (net, "HAR-S x")]
    one = lambda m, k: M[m][k if len(M[m]) > 1 else 0]
    name = lambda a, b: f"pair[{a} | {b}]"
    for a, b in pairs:
        M[name(a, b)] = {k: mix(one(a, k), one(b, (k + 1) % len(RUNS) if a == b else k)) for k in RUNS}
    assert all(np.allclose(M[name(net, LGBM)][k].pred, M[v.hyb][k].pred) for k in RUNS)
    keep = [x for ab in pairs for x in ab] + [name(a, b) for a, b in pairs] + [v.hyb, v.other_hyb, "HAR-S x", TRF]
    per_run, per_bar = V4._losses({n: M[n] for n in dict.fromkeys(keep)}, caps())
    te = lambda f: f.index.get_level_values("split") == "test"
    rows = []
    for a, b in pairs:
        fa = lambda k: M[a][k if len(M[a]) > 1 else 0]
        fb = lambda k: M[b][((k + 1) % len(RUNS) if a == b else k) if len(M[b]) > 1 else 0]
        cc = [float(np.corrcoef((fa(k).pred - fa(k).y)[te(fa(k))], (fb(k).reindex(fa(k).index).pred - fa(k).y)[te(fa(k))])[0, 1]) for k in RUNS]
        n = name(a, b)
        best = min(per_run[a].mse.mean(), per_run[b].mse.mean())
        r = dict(a=a, b=b, mse_mean=per_run[n].mse.mean(), mse_sd=sd(per_run[n].mse), qlike_mean=per_run[n].qlike.mean(), qlike_sd=sd(per_run[n].qlike),
                 mse_a=per_run[a].mse.mean(), mse_b=per_run[b].mse.mean(), corr=np.mean(cc), gain_pct=100 * (per_run[n].mse.mean() / best - 1))
        if (a, b) != (net, LGBM):
            r.update({"hyb_" + c: x for c, x in compare(per_run, per_bar, n, v.hyb).items()})
        rows.append(r)
    fold_of = M[P3.TWIN][0][M[P3.TWIN][0].index.get_level_values("split") == "test"].reset_index().groupby("time").fold.first()
    t = pd.DataFrame({"fold": fold_of})
    for col, (a, b) in {"hybrid_lgbm": (v.hyb, LGBM), "hybrid_net": (v.hyb, net), "hybrid_harsx": (v.hyb, "HAR-S x"), "hybrid_transformer": (v.hyb, TRF),
                        "hybrid_other": (v.hyb, v.other_hyb)}.items():
        for loss in ("mse", "qlike"):
            t[f"{col}_{loss}"] = (per_bar[a][loss] - per_bar[b][loss]).unstack("dataset").mean(axis=1)
    return {"partners": (pd.DataFrame(rows), ["a", "b"]), "time": (t.reset_index(), ["time"])}


def inputs(v) -> dict:
    """paper_v4_paper.inputs: the hybrid of the version on the eight input sets."""
    import paper_v2_round2 as R
    import paper_v3_inputs as I
    from scipy import stats as st
    har = RU.forecasts(*RU.FIXED["HAR"])[0]
    mse_har, _ = RU.losses(har, caps())
    rows, ens, per = [], {}, {}
    for name, gs, key in I.SETS:
        T = RU.forecasts((gs, v.key), I.NET_RUNS[gs])
        G = RU.forecasts(key, I.LGBM_REF if name == I.REF else [I.LGBM_DIR])
        runs = {k: T[k].assign(pred=(T[k].pred + G[k].reindex(T[k].index).pred) / 2) for k in RUNS}
        per[name] = np.array([RU.losses(runs[k], caps()) for k in RUNS])
        ens[name] = runs[0].assign(pred=np.mean([runs[k].reindex(runs[0].index).pred for k in RUNS], axis=0))
        rows.append(dict(model=v.hyb, inputs=name, runs=len(RUNS), mse_mean=per[name][:, 0].mean(), mse_sd=sd(per[name][:, 0]),
                         mse_har=per[name][:, 0].mean() / mse_har, qlike_mean=per[name][:, 1].mean(), qlike_sd=sd(per[name][:, 1])))
    t = pd.DataFrame(rows)
    ref = pd.read_csv(os.path.join(CSV, "v3_paper_models.csv")).set_index("model").loc[v.hyb]
    assert np.isclose(t.loc[t.inputs == I.REF, "mse_mean"].iloc[0], ref.mse_mean, rtol=1e-9)

    def long(name: str) -> pd.DataFrame:
        e = ens[name][ens[name].index.get_level_values("split") == "test"].reset_index()
        return pd.DataFrame({"dataset": e.dataset, "task": "range", "groupset": "IN", "model": name, "seed": 0, "fold": e.fold, "time": e.time, "y": e.y, "pred": e.pred})

    store = R.Store([long(name) for name in ens])
    for i, r in t.iterrows():
        if r.inputs == I.REF:
            continue
        x = store.pair(("IN", r.inputs), ("IN", I.REF))
        t.loc[i, ["vs_ref", "vs_ref_lo", "vs_ref_hi", "vs_ref_better", "p_ref"]] = [x["ratio"], x["ci_lo"], x["ci_hi"], x["better"], x["p"]]
        t.loc[i, "diff_pct"] = 100 * (r.mse_mean / t.loc[t.inputs == I.REF, "mse_mean"].iloc[0] - 1)
        t.loc[i, "p_runs"] = st.ttest_ind(per[r.inputs][:, 0], per[I.REF][:, 0], equal_var=False).pvalue
    return {"inputs": (V4._holm(t, ["model"], ["p_ref"]), ["inputs"])}


def weights(v) -> dict:
    """Fitted weights and stacking coefficients of the three-model schemes with the network of the version (paper_v4_paper.extras)."""
    import paper_v3_triple as TR
    M = M0()
    T, G = M[v.net], M[LGBM]
    rows = []
    for tag, third in (("HAR", M["HAR"][0]), ("HAR-S x", M["HAR-S x"][0])):
        for k in RUNS:
            a = T[k]
            x = pd.DataFrame({"y": a.y, "h": third.reindex(a.index).pred, "g": G[k].reindex(a.index).pred, "n": a.pred})
            for scheme, fn in (("weighted", TR.blend_w), ("stacking", TR.stack)):
                rows += [dict(third=tag, scheme=scheme, run=k, **r) for r in fn(x)[1]]
    return {"triple_weights": (pd.DataFrame(rows), ["third", "scheme", "run", "fold"])}


def example(v, rate: float = 0.05) -> dict:
    """The 95% limits of HAR and of the hybrid of the version (run 0) for BTC in the last test block (paper_v4_paper.limits)."""
    M, frames = M0(), {}
    for name in ("HAR", v.hyb):
        d = M[name][0].reset_index()
        val, test = d[d.split == "val"], d[d.split == "test"].copy()
        c = (val.y - val.pred).groupby([val.dataset, val.fold]).quantile(1 - rate).rename("c")
        test = test.join(c, on=["dataset", "fold"])
        test["limit"] = test.pred + test.c
        frames[name] = test[["time", "dataset", "fold", "y", "pred", "limit"]]
    ex = frames["HAR"][frames["HAR"].dataset.str.startswith("btc")].set_index("time")[["fold", "y", "limit"]].rename(columns={"limit": "limit_har"})
    hy = frames[v.hyb][frames[v.hyb].dataset.str.startswith("btc")].set_index("time")
    ex["pred_hybrid"], ex["limit_hybrid"] = hy.pred, hy.limit
    return {"limits_example": (ex[ex.fold == ex.fold.max()].reset_index(), ["time"])}


# ============================================================================================ driver
ANALYSES = {"main": main, "averages": averages, "regimes": regimes, "encompass": encompass, "rv": rv, "views": views, "extra": extra, "sets": sets, "flat": flat,
            "tuned": tuned, "partners": partners, "inputs": inputs, "weights": weights, "example": example, "twin": twin}
# table of this file -> (table of v4, columns of the v4 table that are named differently there)
V4_NAME = {"main": "v4_r2_main.csv", "averages": "v4_r2_averages.csv", "regimes": "v4_r2_regimes.csv", "encompass": "v4_r2_encompass.csv", "rv": "v4_r2_rv.csv",
           "views": "v4_r3_views.csv", "gains": "v4_r4_gains.csv", "cost": "v4_r3_cost.csv", "variance": "v4_r3_variance.csv", "levels": "v4_r3_limits.csv",
           "linear": "v4_r3_linear.csv", "sets_averages": "v4_r4_averages.csv", "sets_overlap": "v4_r4_overlap.csv", "sets_bandwidth": "v4_r4_bandwidth.csv",
           "sets_print": "v4_r4_print.csv", "flat": "v4_r4_flat.csv", "tuned": "v4_tuned.csv", "partners": "v4_partners.csv", "time": "v4_time.csv",
           "inputs": "v4_inputs.csv", "triple_weights": "v4_triple_weights.csv", "limits_example": "v4_limits_example.csv"}
RENAME = {"time": {"hybrid_other_mse": "hybrid_quaternion_mse", "hybrid_other_qlike": "hybrid_quaternion_qlike"}}


def _same(name: str, new: pd.DataFrame, keys: list[str]) -> str:
    """Require that the table computed with the real network equals the table of v4 on their common rows and columns."""
    old = pd.read_csv(os.path.join(CSV, V4_NAME[name]))
    new = new.rename(columns=RENAME.get(name, {}))
    for k in keys:
        if "time" in k or new[k].dtype.kind == "M":
            new[k], old[k] = pd.to_datetime(new[k], utc=True), pd.to_datetime(old[k], utc=True)
        else:
            new[k], old[k] = new[k].astype(str), old[k].astype(str)
    a, b = new.set_index(keys), old.set_index(keys)
    assert a.index.is_unique and b.index.is_unique, name
    common = b.index.intersection(a.index)
    assert len(common) >= 0.5 * len(b) and len(common) > 0, (name, len(common), len(b))
    cols = [c for c in b.columns if c in a.columns]
    for c in cols:
        x, y = a.loc[common, c], b.loc[common, c]
        if y.dtype.kind in "fiub":
            assert np.allclose(x.astype(float), y.astype(float), rtol=1e-6, atol=1e-9, equal_nan=True), (name, c, float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))))
        else:
            assert (x.fillna("").astype(str) == y.fillna("").astype(str)).all(), (name, c)
    return f"{name}: equal to {V4_NAME[name]} on {len(common)} of {len(b)} rows and {len(cols)} columns"


def run(names: list[str], check: bool):
    v = RV if check else QV
    for n in names:
        if check and n == "twin":
            continue
        for name, (t, keys) in ANALYSES[n](v).items():
            if check:
                if name in V4_NAME:
                    print(_same(name, t, keys), flush=True)
            else:
                t.to_csv(os.path.join(CSV, f"v5_{name}.csv"), index=False)
                print(f"== v5_{name}.csv")
                print(t.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    check = args[:1] == ["check"]
    todo = [a for a in args if a not in ("check", "all")] or list(ANALYSES)
    run(todo, check)
