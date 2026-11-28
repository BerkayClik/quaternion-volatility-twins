"""
Second round of analyses for manuscript v4, after the simulated review of 2026-11-23
(docs/paper/review/v4_round1/reviewer_report.md). All of it is computed from the saved forecasts of the 15 coins of the
paper; no model is trained here.

Plan (2026-11-23, fixed before any number below is computed; committed before the first run of this file). Known
before: every table of manuscript v4 (tables/v3_paper_*.csv, v4_*.csv), and the reviewer's own back-of-the-envelope
figure that a plain average of H-LSTM, LightGBM and the Transformer would have an MSE of about 0.182. "The hybrid" is
H-LSTM + LightGBM. Tests and rule: paper_v3_tests (Welch over the five runs, one-sample where one side has a single
value; Diebold-Mariano over time on the loss difference per bar averaged over runs and coins; better or worse only
if both p < 0.05).

  averages   tables/v4_r2_averages.csv, each against the hybrid, MSE and QLIKE:
             named   plain averages of three and more models: H-LSTM + LightGBM + Transformer; H-QLSTM + LightGBM +
                     Transformer; H-LSTM + H-QLSTM + LightGBM; these four together; the three with PatchTST; all
                     eleven single models.
             top-k   k = 1, ..., 5: per run and fold the k single models with the lowest MSE on the validation block
                     (of the eleven single models of the main table), plain average.
             pair    per run and fold the pair of single models whose plain average has the lowest MSE on the
                     validation block (55 pairs): the choice of the pair as a part of the evaluated procedure.
                     tables/v4_r2_selected.csv lists the choices.
             cost    averages of the five runs of one model (H-LSTM, H-QLSTM, LightGBM, Transformer): five times the
                     training cost of one fit; and the hybrid of the two five-run averages.
  coverage   tables/v4_r2_coverage.csv: for the 95% range limits of paper_v3_paper.limits and every model of the main
             table, per run and coin (3,240 bars) the tests of Kupiec (unconditional coverage), Christoffersen
             (independence of the exceedances) and their sum (conditional coverage); number of coins that reject at
             5% (mean over the runs), and the share of exceedances that follow an exceedance.
             tables/v4_r2_regimes.csv: tick loss in three volatility regimes (terciles of the HAR forecast, with the
             limits of the terciles taken from the validation block of the coin and fold), the hybrid against HAR,
             HAR-S x, LightGBM and the H-LSTM with the two tests.
  encompass  tables/v4_r2_encompass.csv: forecast-encompassing tests (Harvey, Leybourne and Newbold 1998): model a
             encompasses model b if the mean of e_a (e_a - e_b) is zero; tested over time on the series averaged over
             coins and runs, with the Newey-West variance of the paper. Pairs: LightGBM and H-LSTM in both
             directions, LightGBM and H-QLSTM in both directions, and the hybrid against LightGBM, H-LSTM, the
             Transformer and HAR-S x (does the hybrid leave anything to the other model).
  rv         tables/v4_r2_rv.csv: evaluation against the realised variance of the target bar from one-minute returns
             (data/binance_1m, as paper_review_checks.rv_variance: bars with at least 90% of their minutes; the
             variance forecast is exp(2 forecast) times the ratio of the mean realised variance to the mean of
             exp(2 forecast) on the validation block of the coin and fold). Losses: QLIKE and the squared error of
             the log variance. Models: HAR, HAR-S x, LightGBM, Transformer, the two versions of the network and the
             two hybrids, five runs; the hybrid against each with the two tests.
  controls   tables/v4_r2_controls.csv: the design controls of tables/v3_grid.csv for the version with real layers:
             each control with real layers against the reference network with real layers (averages of three runs,
             Diebold-Mariano test and 90% interval, as in paper_v3_tables.grid).

    .venv/bin/python -W ignore experiments/paper_v4_round2.py [averages|coverage|encompass|rv|controls]
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_paper as P3  # noqa: E402
import paper_v3_runs as RU  # noqa: E402
from paper_v3_paper import LGBM, compare  # noqa: E402

CSV, RUNS = RU.OUT, RU.RUNS
Q_NET, R_NET, Q_HYB, R_HYB = P3.NET, P3.TWIN, P3.HYBRID, P3.TWIN_HYBRID
SINGLES = ["HAR", "HAR-S x", LGBM, "DLinear", "CNN, 6 convolutions", "QCNN, 6 convolutions", "PatchTST", "iTransformer", "Transformer", Q_NET, R_NET]
MAIN = SINGLES + [Q_HYB, R_HYB]
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200)


def _two_tests(o: np.ndarray, x: np.ndarray, o_bar: pd.Series, x_bar: pd.Series) -> dict:
    """The two tests of the paper for a loss: runs of ours (o) and of the other model (x), and their losses per bar."""
    from scipy import stats as st
    from src.paper.evaluate import diebold_mariano
    from paper_v3_tests import verdict
    if len(o) > 1 and len(x) > 1:
        p_t = st.ttest_ind(o, x, equal_var=False).pvalue
    else:
        p_t = st.ttest_1samp(o if len(o) > 1 else x, (x if len(o) > 1 else o)[0]).pvalue
    d = (o_bar - x_bar).unstack("dataset")
    series = d.mean(axis=1).dropna().sort_index().to_numpy()
    _, p_dm = diebold_mariano(series, np.zeros(len(series)))
    return dict(diff_pct=100 * (o.mean() / x.mean() - 1), p_runs=p_t, p_time=p_dm, coins_better=int((d.mean(axis=0) < 0).sum()),
                verdict=verdict(o.mean() - x.mean(), p_t, p_dm))


# ============================================================================================ averages
def averages():
    import paper_v2_round2 as R
    from paper_v4_paper import _losses
    M, _ = P3.build()
    one = lambda name, k: M[name][k if len(M[name]) > 1 else 0]
    ref = M[R_NET][0].index
    fold = ref.get_level_values("fold").to_numpy()
    is_val = ref.get_level_values("split").to_numpy() == "val"
    y = M[R_NET][0].y.to_numpy()
    P = {k: pd.DataFrame({n: one(n, k).reindex(ref).pred.to_numpy() for n in SINGLES}, index=ref) for k in RUNS}
    assert all(not p.isna().any().any() for p in P.values())
    frame = lambda pred: pd.DataFrame({"y": y, "pred": np.asarray(pred)}, index=ref)
    new, chosen = {}, []

    named = {"H-LSTM + LightGBM + Transformer": [R_NET, LGBM, "Transformer"], "H-QLSTM + LightGBM + Transformer": [Q_NET, LGBM, "Transformer"],
             "H-LSTM + H-QLSTM + LightGBM": [R_NET, Q_NET, LGBM], "H-LSTM + H-QLSTM + LightGBM + Transformer": [R_NET, Q_NET, LGBM, "Transformer"],
             "H-LSTM + LightGBM + Transformer + PatchTST": [R_NET, LGBM, "Transformer", "PatchTST"], "All eleven single models": SINGLES}
    for name, members in named.items():
        new[("named", name)] = {k: frame(P[k][members].mean(axis=1)) for k in RUNS}

    val_mse = lambda pred, f: float(np.mean((pred[is_val & (fold == f)] - y[is_val & (fold == f)]) ** 2))
    for kk in (1, 2, 3, 4, 5):
        out = {}
        for k in RUNS:
            pred = np.full(len(ref), np.nan)
            for f in np.unique(fold):
                rank = sorted(SINGLES, key=lambda n: val_mse(P[k][n].to_numpy(), f))[:kk]
                pred[fold == f] = P[k].loc[fold == f, rank].mean(axis=1)
                chosen.append(dict(scheme=f"top-{kk}", run=k, fold=int(f), members=" | ".join(sorted(rank))))
            out[k] = frame(pred)
        new[("top-k", f"Best {kk} on the validation block")] = out
    out = {}
    for k in RUNS:
        pred = np.full(len(ref), np.nan)
        for f in np.unique(fold):
            a, b = min(itertools.combinations(SINGLES, 2), key=lambda ab: val_mse(P[k][list(ab)].mean(axis=1).to_numpy(), f))
            pred[fold == f] = P[k].loc[fold == f, [a, b]].mean(axis=1)
            chosen.append(dict(scheme="pair", run=k, fold=int(f), members=" | ".join(sorted([a, b]))))
        out[k] = frame(pred)
    new[("pair", "Best pair on the validation block")] = out

    mean5 = {n: np.mean([P[k][n].to_numpy() for k in RUNS], axis=0) for n in (R_NET, Q_NET, LGBM, "Transformer")}
    for n, label in ((R_NET, "H-LSTM"), (Q_NET, "H-QLSTM"), (LGBM, "LightGBM"), ("Transformer", "Transformer")):
        new[("cost", f"Five runs of {label}")] = {0: frame(mean5[n])}
    new[("cost", "Five runs of H-LSTM + five runs of LightGBM")] = {0: frame((mean5[R_NET] + mean5[LGBM]) / 2)}

    allm = {R_HYB: M[R_HYB], LGBM: M[LGBM], **{name: v for (_, name), v in new.items()}}
    per_run, per_bar = _losses(allm, R.train_caps())
    rows = []
    for (kind, name) in new:
        p = per_run[name]
        rows.append(dict(kind=kind, name=name, runs=len(p), mse_mean=p.mse.mean(), mse_sd=sd(p.mse), qlike_mean=p.qlike.mean(), qlike_sd=sd(p.qlike),
                         **{"hyb_" + c: v for c, v in compare(per_run, per_bar, name, R_HYB).items()},
                         **{"lgbm_" + c: v for c, v in compare(per_run, per_bar, name, LGBM).items() if c.startswith("mse_")}))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v4_r2_averages.csv"), index=False)
    c = pd.DataFrame(chosen)
    c.to_csv(os.path.join(CSV, "v4_r2_selected.csv"), index=False)
    print(f"hybrid: {per_run[R_HYB].mse.mean():.4f}")
    print(t[["kind", "name", "runs", "mse_mean", "mse_sd", "hyb_mse_diff_pct", "hyb_mse_diff_lo", "hyb_mse_diff_hi", "hyb_mse_p_runs", "hyb_mse_p_time",
             "hyb_mse_coins_better", "hyb_mse_verdict", "qlike_mean", "hyb_qlike_diff_pct", "hyb_qlike_verdict", "lgbm_mse_diff_pct"]].round(4).to_string(index=False))
    print(c.groupby(["scheme", "members"]).size().rename("fits").reset_index().sort_values(["scheme", "fits"], ascending=[True, False]).to_string(index=False))


# ============================================================================================ coverage
def _limits(M: dict, names: list[str], rate: float = 0.05) -> dict:
    """name -> list over runs of the test frame with the limit (paper_v3_paper.limits)."""
    out = {}
    for name in names:
        runs = []
        for f in M[name].values():
            d = f.reset_index()
            val, test = d[d.split == "val"], d[d.split == "test"].copy()
            c = (val.y - val.pred).groupby([val.dataset, val.fold]).quantile(1 - rate).rename("c")
            test = test.join(c, on=["dataset", "fold"])
            u = test.y - (test.pred + test.c)
            test["hit"] = (u > 0).astype(int)
            test["tick"] = u * ((1 - rate) - (u < 0))
            runs.append(test.sort_values(["dataset", "time"]).reset_index(drop=True))
        out[name] = runs
    return out


def _lr_tests(hit: np.ndarray, p: float = 0.05) -> tuple[float, float, float]:
    """p-values of the Kupiec, the Christoffersen independence and the conditional coverage test."""
    from scipy.stats import chi2
    ll = lambda k, n, q: 0.0 if n == 0 else (k * np.log(q) if k else 0.0) + ((n - k) * np.log(1 - q) if n - k else 0.0)
    n, x = len(hit), int(hit.sum())
    uc = -2 * (ll(x, n, p) - ll(x, n, x / n))
    a, b = hit[:-1], hit[1:]
    n00, n01, n10, n11 = (int(((a == i) & (b == j)).sum()) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    pi = (n01 + n11) / (n - 1)
    pi0, pi1 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    ind = -2 * (ll(n01 + n11, n - 1, pi) - ll(n01, n00 + n01, pi0) - ll(n11, n10 + n11, pi1))
    return float(chi2.sf(uc, 1)), float(chi2.sf(ind, 1)), float(chi2.sf(uc + ind, 2))


def coverage():
    M, _ = P3.build()
    L = _limits(M, MAIN)
    rows = []
    for name in MAIN:
        per = []
        for test in L[name]:
            ps = np.array([_lr_tests(g.hit.to_numpy()) for _, g in test.groupby("dataset")])
            prev = test.groupby("dataset").hit.shift(1)
            per.append(dict(exceed=test.hit.mean(), uc=(ps[:, 0] < 0.05).sum(), ind=(ps[:, 1] < 0.05).sum(), cc=(ps[:, 2] < 0.05).sum(),
                            after_hit=test.hit[prev == 1].mean(), after_none=test.hit[prev == 0].mean()))
        per = pd.DataFrame(per)
        rows.append(dict(model=name, runs=len(per), exceed=per.exceed.mean(), coins_reject_uc=per.uc.mean(), coins_reject_ind=per.ind.mean(),
                         coins_reject_cc=per.cc.mean(), exceed_after_exceedance=per.after_hit.mean(), exceed_after_none=per.after_none.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v4_r2_coverage.csv"), index=False)
    print(t.round(4).to_string(index=False))

    # regimes: terciles of the HAR forecast, limits from the validation block of the coin and fold
    har = M["HAR"][0].reset_index()
    q = har[har.split == "val"].groupby(["dataset", "fold"]).pred.quantile([1 / 3, 2 / 3]).unstack().rename(columns={1 / 3: "q1", 2 / 3: "q2"})
    ht = har[har.split == "test"].join(q, on=["dataset", "fold"])
    ht["regime"] = np.where(ht.pred <= ht.q1, "low", np.where(ht.pred <= ht.q2, "middle", "high"))
    regime = ht.set_index(["dataset", "time"]).regime
    in_regime = lambda test, reg: test[test.set_index(["dataset", "time"]).index.map(regime).to_numpy() == reg]
    rows = []
    for reg in ("low", "middle", "high"):
        stat = {}
        for name in ("HAR", "HAR-S x", LGBM, R_NET, Q_HYB, R_HYB):
            parts = [in_regime(test, reg) for test in L[name]]
            bars = [x.set_index(["time", "dataset"]).tick.sort_index() for x in parts]
            stat[name] = dict(runs=np.array([x.tick.mean() for x in parts]), bar=sum(bars) / len(bars), n=len(parts[0]),
                              exceed=float(np.mean([x.hit.mean() for x in parts])))
        ours = stat[R_HYB]
        for name, s_ in stat.items():
            r = dict(regime=reg, bars=s_["n"], model=name, exceed=s_["exceed"], tick_mean=s_["runs"].mean(), tick_sd=sd(s_["runs"]))
            if name != R_HYB:
                r.update(_two_tests(ours["runs"], s_["runs"], ours["bar"], s_["bar"]))
            rows.append(r)
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(CSV, "v4_r2_regimes.csv"), index=False)
    print(g.round(4).to_string(index=False))


# ============================================================================================ encompass
def encompass():
    from src.paper.evaluate import diebold_mariano
    M, _ = P3.build()
    test = lambda f: f[f.index.get_level_values("split") == "test"]
    err = {n: {k: (lambda f: (f.y - f.pred))(test(M[n][k if len(M[n]) > 1 else 0])) for k in RUNS} for n in (LGBM, R_NET, Q_NET, "Transformer", "HAR-S x", R_HYB)}
    pairs = [(LGBM, R_NET), (R_NET, LGBM), (LGBM, Q_NET), (Q_NET, LGBM), (R_HYB, LGBM), (R_HYB, R_NET), (R_HYB, "Transformer"), (R_HYB, "HAR-S x")]
    rows = []
    for a, b in pairs:
        d = sum(err[a][k] * (err[a][k] - err[b][k].reindex(err[a][k].index)) for k in RUNS) / len(RUNS)
        lam = float(np.mean([(err[a][k] * (err[a][k] - err[b][k].reindex(err[a][k].index))).sum() / ((err[a][k] - err[b][k].reindex(err[a][k].index)) ** 2).sum() for k in RUNS]))
        series = d.droplevel(["fold", "split"]).unstack("dataset").mean(axis=1).sort_index().to_numpy()
        stat, p = diebold_mariano(series, np.zeros(len(series)))
        rows.append(dict(model=a, other=b, weight_on_other=lam, mean=float(series.mean()), statistic=float(stat), p=float(p),
                         encompasses=bool(p >= 0.05 or series.mean() < 0)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v4_r2_encompass.csv"), index=False)
    print(t.round(4).to_string(index=False))


# ============================================================================================ rv
def rv():
    M, _ = P3.build()
    names = ["HAR", "HAR-S x", LGBM, "Transformer", Q_NET, R_NET, Q_HYB, R_HYB]
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
            v = d[d.split == "val"].groupby(["dataset", "fold"]).agg(rv=("rv", "mean"), raw=("raw", "mean"))
            te = d[d.split == "test"].join((v.rv / v.raw).rename("c"), on=["dataset", "fold"])
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
            if name != R_HYB:
                r.update({f"{loss}_{c}": v for c, v in _two_tests(per_run[R_HYB][loss].to_numpy(), x, per_bar[R_HYB][loss], per_bar[name][loss]).items()})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v4_r2_rv.csv"), index=False)
    print(t.round(4).to_string(index=False))


# ============================================================================================ controls
def controls():
    import paper_v2_round2 as R
    import paper_v3_tables as T
    base = pd.concat([T._run(T.A_RUN, [T.A, T.AR]), T._run(T.HAR_RUN, [R.HAR], cells_needed=60)], ignore_index=True)
    rows = []
    for block, label, pattern, q, r in T.GRID:
        if q is None or r is None:
            continue
        x = T._run(pattern, [q, r], "@c")
        if x is None:
            print("not available:", block, "/", label)
            continue
        tag = lambda k: (k[0], k[1] + "@c")
        st = R.Store([base, x])
        v, w = st.pair(tag(r), T.AR), st.pair(tag(q), T.A)
        rows.append(dict(block=block, control=label, real=st.vs_har(tag(r)), real_vs_ref=v["ratio"], real_lo=v["ci_lo"], real_hi=v["ci_hi"], real_p=v["p"],
                         real_better=v["better"], quaternion_vs_ref=w["ratio"], quaternion_p=w["p"]))
    t = pd.DataFrame(rows)
    old = pd.read_csv(os.path.join(CSV, "v3_grid.csv")).set_index("control")
    assert np.allclose(t.set_index("control").quaternion_vs_ref, old.loc[t.control, "vs_A"], rtol=1e-9)
    t.to_csv(os.path.join(CSV, "v4_r2_controls.csv"), index=False)
    print(t.round(4).to_string(index=False))


# ============================================================================================ main table, extended
EXTRA_SINGLE = ["GARCH(1,1)", "GJR-GARCH", "CARR", "Linear, pooled", "HGB"]
EXTRA_AVG = ["Five runs of H-LSTM", "LightGBM + HGB", "H-LSTM + HGB", "Transformer + LightGBM", "H-LSTM + Transformer", "Five runs of H-LSTM + LightGBM",
             "H-LSTM + LightGBM + Transformer"]


def main_ext():
    """tables/v4_r2_main.csv: one row per model of the extended main table (the 13 models of manuscript v4, the further
    benchmarks of paper_v4_bench and the averages of the second round), all computed by one routine: MSE and QLIKE over
    five runs, the hybrid against the model with the two tests (Holm over all rows), and the model confidence set
    (90%, as in paper_v3_paper.stats) over all rows. The confidence set over the extended list had no plan fixed in
    advance and is descriptive."""
    import paper_v2_round2 as R
    from arch.bootstrap import MCS
    from paper_v4_paper import _holm, _losses
    M, _ = P3.build()
    get = RU.forecasts
    ref = M[R_NET][0].index
    y = M[R_NET][0].y.to_numpy()
    frame = lambda pred: pd.DataFrame({"y": y, "pred": np.asarray(pred)}, index=ref)
    pred = lambda f: f.reindex(ref).pred.to_numpy()
    X = {n: M[n] for n in MAIN}
    X["GARCH(1,1)"] = {0: get(("P", "garch"), ["results/v4bench/garch"])[0].reindex(ref)}
    X["GJR-GARCH"] = {0: get(("P", "gjr"), ["results/v4bench/garch"])[0].reindex(ref)}
    X["HGB"] = {k: f.reindex(ref) for k, f in get(("PM", "hgbsl-pool"), ["results/v4bench/hgb"]).items()}
    X["CARR"] = {0: get(("P", "carr"), ["results/v4views/linear"])[0].reindex(ref)}                      # third round, point 4
    X["Linear, pooled"] = {0: get(("PM", "linear-pool"), ["results/v4views/linear"])[0].reindex(ref)}
    avg = lambda *names: {k: frame(np.mean([pred(X[n][k]) for n in names], axis=0)) for k in RUNS}
    X["Five runs of H-LSTM"] = {0: frame(np.mean([pred(M[R_NET][k]) for k in RUNS], axis=0))}
    X["LightGBM + HGB"], X["H-LSTM + HGB"] = avg(LGBM, "HGB"), avg(R_NET, "HGB")
    X["Transformer + LightGBM"], X["H-LSTM + Transformer"] = avg("Transformer", LGBM), avg(R_NET, "Transformer")
    X["H-LSTM + LightGBM + Transformer"] = avg(R_NET, LGBM, "Transformer")
    mean5 = np.mean([pred(M[R_NET][k]) for k in RUNS], axis=0)
    X["Five runs of H-LSTM + LightGBM"] = {k: frame((mean5 + pred(M[LGBM][k])) / 2) for k in RUNS}
    per_run, per_bar = _losses(X, R.train_caps())
    har = per_run["HAR"].mean()
    rows = []
    for name in X:
        p = per_run[name]
        r = dict(ours=R_HYB, model=name, runs=len(p), mse_mean=p.mse.mean(), mse_sd=sd(p.mse), mse_har=p.mse.mean() / har.mse, qlike_mean=p.qlike.mean(),
                 qlike_sd=sd(p.qlike), qlike_median=per_bar[name].qlike.median())
        if name != R_HYB:
            r.update(compare(per_run, per_bar, R_HYB, name))
        rows.append(r)
    t = _holm(pd.DataFrame(rows), ["ours"], ["mse_p_runs", "mse_p_time", "qlike_p_runs", "qlike_p_time"])
    L = pd.DataFrame({n: per_bar[n].mse.unstack("dataset").mean(axis=1) for n in X}).dropna()
    m = MCS(L.to_numpy(), size=0.10, reps=2000, block_size=14, seed=0)
    m.compute()
    pv = m.pvalues.squeeze()
    t["mcs_p"] = [float(pv.loc[i]) for i in range(len(L.columns))]
    old = pd.read_csv(os.path.join(CSV, "v4_tests.csv"))
    old = old[old.ours == R_HYB].set_index("other")
    new = t.set_index("model")
    assert np.allclose(old.mse_diff_pct, new.loc[old.index, "mse_diff_pct"], rtol=1e-6) and np.allclose(old.mse_p_time, new.loc[old.index, "mse_p_time"], rtol=1e-6, atol=1e-12)
    t.to_csv(os.path.join(CSV, "v4_r2_main.csv"), index=False)
    print(t[["model", "runs", "mse_mean", "mse_sd", "mse_har", "qlike_mean", "mse_diff_pct", "mse_p_runs", "mse_p_time", "mse_coins_better", "mse_verdict",
             "mse_p_runs_holm", "mse_p_time_holm", "qlike_diff_pct", "qlike_p_runs", "qlike_p_time", "qlike_verdict", "mcs_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    fns = {"averages": averages, "coverage": coverage, "encompass": encompass, "rv": rv, "controls": controls, "main": main_ext}
    for a in sys.argv[1:]:
        fns[a]()
