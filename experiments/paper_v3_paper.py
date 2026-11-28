"""
Statistics, tables and figures of manuscript v3 (docs/paper/manuscript_v3: the hybrid of the H-QLSTM and LightGBM),
computed from the saved forecasts and from the tables of the V3 experiments. No model is trained here.

    stats    descriptive statistics that the earlier V3 scripts do not hold (tables/v3_paper_*.csv):
               models   five runs of every model and combination: MSE and QLIKE, mean and standard deviation
               combos   every combination scheme against the plain average of the H-QLSTM and LightGBM and against
                        LightGBM, with the two tests of paper_v3_tests (Welch over runs, Diebold-Mariano over time
                        on the loss difference averaged over runs and coins) and its rule
               corr     correlation of the forecast errors of two models (run k with run k, averaged over the runs;
                        on the diagonal the mean correlation of two runs of the same model)
               weight   MSE of w * network + (1 - w) * LightGBM for w = 0, 0.05, ..., 1 on the validation and on
                        the test blocks (mean over the runs)
               time     loss difference per test bar, averaged over coins and runs (for the cumulative figure)
               coins    MSE per coin (mean over the runs)
               partners plain average of two single models (run k with run k; for two runs of one model, run k
                        with run k + 1): MSE, the correlation of the two errors, the gain over the better member
               mcs      model confidence set (90%) of the models of the main table, on the squared error and on
                        QLIKE averaged over coins and runs (range statistic, block length 14, 2,000 replications)
    limits   tables/v3_paper_limits.csv: the forecasts as 95% range limits (plan in the docstring of `limits`)
    tables   docs/paper/manuscript_v3/tables/*.tex
    figs     docs/paper/manuscript_v3/figures/*.pdf

    .venv/bin/python -W ignore experiments/paper_v3_paper.py [stats|tables|figs]
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_runs as RU  # noqa: E402

CSV, RUNS = RU.OUT, RU.RUNS
PAPER = "docs/paper/manuscript_v3"
HYBRID, TWIN_HYBRID = "H-QLSTM + LightGBM", "H-LSTM + LightGBM"
NET, TWIN, LGBM = "H-QLSTM", "H-LSTM (real twin)", "LightGBM-S, lagged"
MAIN = ["HAR", "HAR-S x", LGBM, "DLinear", "CNN, 6 convolutions", "QCNN, 6 convolutions", "PatchTST", "iTransformer", "Transformer",
        TWIN, NET, TWIN_HYBRID, HYBRID]
# (group, label, name in `build`)
COMBOS = [
    ("H-QLSTM and LightGBM", "Plain average (proposed)", HYBRID),
    ("H-QLSTM and LightGBM", "Weighted average", "W2"),
    ("H-QLSTM and LightGBM", "Linear stacking", "S2"),
    ("H-QLSTM and LightGBM", "Cascade: LightGBM, then H-QLSTM", "Cascade: LightGBM, then H-QLSTM"),
    ("H-QLSTM and HAR", "Plain average with HAR", "A2H"),
    ("H-QLSTM and HAR", "Plain average with HAR-S$\\times$", "A2HS"),
    ("H-QLSTM and HAR", "Cascade: HAR, then H-QLSTM", "Cascade: HAR, then H-QLSTM"),
    ("H-QLSTM and HAR", "Cascade: HAR-S$\\times$, then H-QLSTM", "Cascade: HAR-S x, then H-QLSTM"),
    ("HAR, LightGBM and H-QLSTM", "Plain average", "A3"),
    ("HAR, LightGBM and H-QLSTM", "Weighted average", "W3"),
    ("HAR, LightGBM and H-QLSTM", "Linear stacking", "S3"),
    ("HAR, LightGBM and H-QLSTM", "Cascade: HAR, LightGBM, H-QLSTM", "C3a"),
    ("HAR, LightGBM and H-QLSTM", "Cascade: HAR, H-QLSTM, LightGBM", "C3b"),
    ("HAR-S$\\times$, LightGBM and H-QLSTM", "Plain average", "A3S"),
    ("HAR-S$\\times$, LightGBM and H-QLSTM", "Weighted average", "W3S"),
    ("HAR-S$\\times$, LightGBM and H-QLSTM", "Linear stacking", "S3S"),
]
W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
PARTNERS = [(NET, LGBM), (TWIN, LGBM), ("Transformer", LGBM), ("PatchTST", LGBM), ("iTransformer", LGBM), ("HAR-S x", LGBM), (LGBM, LGBM),
            (NET, NET), (NET, TWIN), (NET, "Transformer"), (NET, "PatchTST"), (NET, "HAR-S x")]
pair_name = lambda a, b: f"avg[{a} | {b}]"


def weighted(n: pd.DataFrame, g: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """w * n + (1 - w) * g with, per fold, the w of W_GRID that has the least MSE on the validation block."""
    out, ws = n.copy(), []
    g = g.reindex(n.index)
    for fold, f in n.groupby(level="fold"):
        va = f.index[f.index.get_level_values("split") == "val"]
        en, eg = (n.loc[va, "pred"] - n.loc[va, "y"]).to_numpy(), (g.loc[va, "pred"] - n.loc[va, "y"]).to_numpy()
        mse = [np.mean((w * en + (1 - w) * eg) ** 2) for w in W_GRID]
        w = float(W_GRID[int(np.argmin(mse))])
        out.loc[f.index, "pred"] = w * n.loc[f.index, "pred"] + (1 - w) * g.loc[f.index, "pred"]
        ws.append(w)
    return out, ws


def build() -> tuple[dict, dict]:
    """name -> {run: frame(y, pred) indexed by RU.IDX} for every model and combination; and the fitted weights."""
    import glob
    import paper_v3_triple as TR
    get = RU.forecasts
    runs = {name: get(key, pats) for name, (key, pats) in RU.SEEDED.items()}
    har, harsx = get(*RU.FIXED["HAR"])[0], get(*RU.FIXED["HAR-S x"])[0]
    A, T, G = runs["H-QLSTM (candidate A)"], runs["H-LSTM (real twin)"], runs[LGBM]
    mix = lambda a, b: a.assign(pred=(a.pred + b.reindex(a.index).pred) / 2)
    M = {"HAR": {0: har}, "HAR-S x": {0: harsx}}
    for name in (LGBM, "DLinear", "CNN, 6 convolutions", "QCNN, 6 convolutions", "PatchTST", "iTransformer", "Transformer",
                 "Cascade: LightGBM, then H-QLSTM", "Cascade: LightGBM, then H-LSTM", "Cascade: HAR, then H-QLSTM", "Cascade: HAR-S x, then H-QLSTM"):
        M[name] = {k: runs[name][k] for k in RUNS}
    M[NET], M[TWIN] = {k: A[k] for k in RUNS}, {k: T[k] for k in RUNS}
    M[HYBRID], M[TWIN_HYBRID] = {k: mix(A[k], G[k]) for k in RUNS}, {k: mix(T[k], G[k]) for k in RUNS}
    weights = {"W2": {}, "W2 twin": {}}
    M["W2"], M["W2 twin"] = {}, {}
    for k in RUNS:
        M["W2"][k], weights["W2"][k] = weighted(A[k], G[k])
        M["W2 twin"][k], weights["W2 twin"][k] = weighted(T[k], G[k])
    M["S2"] = {k: RU.stack(A[k], G[k].reindex(A[k].index)) for k in RUNS}
    M["A2H"], M["A2HS"] = {k: mix(A[k], har) for k in RUNS}, {k: mix(A[k], harsx) for k in RUNS}

    def wide(k: int, third: pd.DataFrame) -> pd.DataFrame:
        a = A[k]
        x = pd.DataFrame({"y": a.y, "h": third.reindex(a.index).pred, "g": G[k].reindex(a.index).pred, "n": a.pred})
        assert x.notna().all().all()
        return x

    for tag, third in (("", har), ("S", harsx)):
        M["A3" + tag] = {k: (lambda x: x[["y"]].assign(pred=x[["h", "g", "n"]].mean(1)))(wide(k, third)) for k in RUNS}
        M["W3" + tag] = {k: TR.blend_w(wide(k, third))[0] for k in RUNS}
        M["S3" + tag] = {k: TR.stack(wide(k, third))[0] for k in RUNS}
    M["C3a"] = get(TR.NET, TR.HGN_RUNS) if glob.glob(TR.HGN_RUNS[0]) else {}
    M["C3b"] = get(TR.HNG, [TR.HNG_DIR])
    for name in ("C3a", "C3b"):
        M[name] = {k: M[name][k] for k in RUNS}
    for a, b in PARTNERS:                      # plain average of two single models; two runs of one model: run k with run k + 1
        one = lambda m, k: M[m][k if len(M[m]) > 1 else 0]
        M[pair_name(a, b)] = {k: mix(one(a, k), one(b, (k + 1) % len(RUNS) if a == b else k)) for k in RUNS}
    return M, weights


def compare(per_run: dict, per_bar: dict, ours: str, other: str) -> dict:
    """The two tests of paper_v3_tests for `ours` against `other`, MSE and QLIKE."""
    from scipy import stats
    from src.paper.evaluate import diebold_mariano
    from paper_v3_tests import verdict
    r = {}
    for loss in ("mse", "qlike"):
        x, y = per_run[ours][loss].to_numpy(), per_run[other][loss].to_numpy()
        if len(x) > 1 and len(y) > 1:
            p_t = stats.ttest_ind(x, y, equal_var=False).pvalue
        else:
            p_t = stats.ttest_1samp(x if len(x) > 1 else y, (y if len(x) > 1 else x)[0]).pvalue
        d = (per_bar[ours][loss] - per_bar[other][loss]).unstack("dataset")
        series = d.mean(axis=1).sort_index().to_numpy()
        dm, p_dm = diebold_mariano(series, np.zeros(len(series)))
        se = abs(series.mean() / dm)
        assert np.isclose(series.mean(), x.mean() - y.mean(), rtol=1e-6, atol=1e-9), (ours, other, loss)
        r.update({f"{loss}_diff_pct": 100 * (x.mean() / y.mean() - 1), f"{loss}_diff_lo": 100 * (series.mean() - 1.645 * se) / y.mean(),
                  f"{loss}_diff_hi": 100 * (series.mean() + 1.645 * se) / y.mean(), f"{loss}_coins_better": int((d.mean(axis=0) < 0).sum()),
                  f"{loss}_p_runs": p_t, f"{loss}_p_time": p_dm, f"{loss}_verdict": verdict(x.mean() - y.mean(), p_t, p_dm)})
    return r


def stats():
    import paper_v2_round2 as R
    from arch.bootstrap import MCS
    from paper_v3_tests import bar_losses
    caps = R.train_caps()
    M, weights = build()
    per_run, per_bar = {}, {}
    for name, by_run in M.items():
        bars = [bar_losses(f, caps) for f in by_run.values()]
        per_run[name] = pd.DataFrame([b.mean() for b in bars])
        per_bar[name] = sum(bars) / len(bars)
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    har = per_run["HAR"].mean()

    # models: every model and combination
    rows = [dict(model=name, runs=len(p), mse_mean=p.mse.mean(), mse_sd=sd(p.mse), mse_har=p.mse.mean() / har.mse,
                 qlike_mean=p.qlike.mean(), qlike_sd=sd(p.qlike), qlike_har=p.qlike.mean() / har.qlike, qlike_median=per_bar[name].qlike.median())
            for name, p in per_run.items() if not name.startswith("avg[")]
    models = pd.DataFrame(rows)
    models.to_csv(os.path.join(CSV, "v3_paper_models.csv"), index=False)
    old = pd.read_csv(os.path.join(CSV, "v3_runs.csv")).set_index("model")
    for new, was in ((HYBRID, "Blending: H-QLSTM + LightGBM"), (NET, "H-QLSTM (candidate A)"), ("S2", "Stacking: H-QLSTM + LightGBM"), (LGBM, LGBM)):
        m = models.set_index("model").loc[new]
        assert np.isclose(m.mse_mean, old.loc[was, "mse_mean"], rtol=1e-9) and np.isclose(m.qlike_mean, old.loc[was, "qlike_mean"], rtol=1e-9), new

    # combos: every scheme against the plain average and against LightGBM
    rows = []
    for group, label, name in COMBOS:
        p = per_run[name]
        r = dict(group=group, scheme=label, name=name, mse_mean=p.mse.mean(), mse_sd=sd(p.mse), qlike_mean=p.qlike.mean(), qlike_sd=sd(p.qlike))
        if name != HYBRID:
            r.update({"avg_" + k: v for k, v in compare(per_run, per_bar, name, HYBRID).items()})
        r.update({"lgbm_" + k: v for k, v in compare(per_run, per_bar, name, LGBM).items()})
        rows.append(r)
    combos = pd.DataFrame(rows)
    combos.to_csv(os.path.join(CSV, "v3_paper_combos.csv"), index=False)
    wtab = pd.DataFrame([dict(scheme=s, run=k, fold=f, weight_on_network=v) for s, by in weights.items() for k, ws in by.items() for f, v in enumerate(ws)])
    wtab.to_csv(os.path.join(CSV, "v3_paper_weights.csv"), index=False)

    # corr: forecast errors on the test blocks
    names = ["HAR", "HAR-S x", LGBM, "PatchTST", "iTransformer", "Transformer", TWIN, NET]
    err = {n: {k: (f.pred - f.y)[f.index.get_level_values("split") == "test"] for k, f in M[n].items()} for n in names}
    rows = []
    for a in names:
        for b in names:
            if a == b:
                ks = list(err[a])
                c = [np.corrcoef(err[a][i], err[a][j].reindex(err[a][i].index))[0, 1] for i in ks for j in ks if i < j]
            else:
                c = [np.corrcoef(err[a][i if len(err[a]) > 1 else 0], err[b][i if len(err[b]) > 1 else 0].reindex(err[a][i if len(err[a]) > 1 else 0].index))[0, 1]
                     for i in RUNS]
            rows.append(dict(a=a, b=b, corr=float(np.mean(c)) if c else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(CSV, "v3_paper_corr.csv"), index=False)

    # weight: MSE of w * network + (1 - w) * LightGBM on the validation and the test blocks
    rows = []
    for label, net in ((NET, M[NET]), (TWIN, M[TWIN])):
        for w in np.round(np.arange(0, 1.0001, 0.05), 2):
            for split in ("val", "test"):
                v = []
                for k in RUNS:
                    n, g = net[k], M[LGBM][k].reindex(net[k].index)
                    m = n.index.get_level_values("split") == split
                    v.append(float(np.mean((w * n.pred[m] + (1 - w) * g.pred[m] - n.y[m]) ** 2)))
                rows.append(dict(network=label, w=w, split=split, mse=np.mean(v), mse_sd=sd(v)))
    pd.DataFrame(rows).to_csv(os.path.join(CSV, "v3_paper_weight.csv"), index=False)

    # time: loss difference per test bar (mean over coins and runs)
    fold_of = M[NET][0][M[NET][0].index.get_level_values("split") == "test"].reset_index().groupby("time").fold.first()
    t = pd.DataFrame({"fold": fold_of})
    for col, (a, b) in {"hybrid_lgbm": (HYBRID, LGBM), "hybrid_net": (HYBRID, NET), "hybrid_harsx": (HYBRID, "HAR-S x"),
                        "hybrid_transformer": (HYBRID, "Transformer"), "net_twin": (NET, TWIN), "hybrid_twinhybrid": (HYBRID, TWIN_HYBRID)}.items():
        for loss in ("mse", "qlike"):
            t[f"{col}_{loss}"] = (per_bar[a][loss] - per_bar[b][loss]).unstack("dataset").mean(axis=1)
    t.to_csv(os.path.join(CSV, "v3_paper_time.csv"))

    # coins: MSE per coin
    c = pd.DataFrame({n: per_bar[n].mse.unstack("dataset").mean(axis=0) for n in ("HAR", "HAR-S x", LGBM, TWIN, NET, TWIN_HYBRID, HYBRID)})
    c.to_csv(os.path.join(CSV, "v3_paper_coins.csv"))

    # partners: plain average of two single models, with the two tests against the proposed hybrid
    rows = []
    for a, b in PARTNERS:
        name = pair_name(a, b)
        fa = lambda k: M[a][k if len(M[a]) > 1 else 0]
        fb = lambda k: M[b][((k + 1) % len(RUNS) if a == b else k) if len(M[b]) > 1 else 0]
        te = lambda f: f.index.get_level_values("split") == "test"
        cc = [float(np.corrcoef((fa(k).pred - fa(k).y)[te(fa(k))], (fb(k).reindex(fa(k).index).pred - fa(k).y)[te(fa(k))])[0, 1]) for k in RUNS]
        best = min(per_run[a].mse.mean(), per_run[b].mse.mean())
        r = dict(a=a, b=b, mse_mean=per_run[name].mse.mean(), mse_sd=sd(per_run[name].mse), qlike_mean=per_run[name].qlike.mean(),
                 qlike_sd=sd(per_run[name].qlike), mse_a=per_run[a].mse.mean(), mse_b=per_run[b].mse.mean(), corr=np.mean(cc),
                 gain_pct=100 * (per_run[name].mse.mean() / best - 1))
        if (a, b) != (NET, LGBM):
            r.update({"hyb_" + k: v for k, v in compare(per_run, per_bar, name, HYBRID).items()})
        else:
            assert np.isclose(r["mse_mean"], per_run[HYBRID].mse.mean(), rtol=1e-12)
        rows.append(r)
    partners = pd.DataFrame(rows)
    partners.to_csv(os.path.join(CSV, "v3_paper_partners.csv"), index=False)

    # mcs: the models of the main table
    rows = []
    for loss in ("mse", "qlike"):
        L = pd.DataFrame({n: per_bar[n][loss].unstack("dataset").mean(axis=1) for n in MAIN}).dropna()
        m = MCS(L.to_numpy(), size=0.10, reps=2000, block_size=14, seed=0)
        m.compute()
        pv = m.pvalues.squeeze()
        rows += [dict(loss=loss, model=n, mean=L[n].mean(), mcs_p=float(pv.loc[i]), in_set=bool(pv.loc[i] >= 0.10)) for i, n in enumerate(L.columns)]
    mcs = pd.DataFrame(rows)
    mcs.to_csv(os.path.join(CSV, "v3_paper_mcs.csv"), index=False)

    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200)
    print(models.round(4).to_string(index=False))
    print(combos[["scheme", "mse_mean", "mse_sd", "avg_mse_diff_pct", "avg_mse_p_runs", "avg_mse_p_time", "avg_mse_verdict", "qlike_mean", "qlike_sd",
                  "avg_qlike_diff_pct", "avg_qlike_p_runs", "avg_qlike_p_time", "avg_qlike_verdict", "lgbm_mse_diff_pct", "lgbm_mse_verdict",
                  "lgbm_qlike_diff_pct", "lgbm_qlike_verdict"]].round(4).to_string(index=False))
    print(wtab.groupby(["scheme", "fold"]).weight_on_network.agg(["mean", "min", "max"]).round(2).to_string())
    print(pd.read_csv(os.path.join(CSV, "v3_paper_corr.csv")).pivot(index="a", columns="b", values="corr").loc[names, names].round(3).to_string())
    print(pd.read_csv(os.path.join(CSV, "v3_paper_weight.csv")).pivot(index="w", columns=["network", "split"], values="mse").round(5).to_string())
    print((c.div(c["HAR"], axis=0)).round(4).to_string())
    print(partners[["a", "b", "mse_mean", "mse_sd", "qlike_mean", "corr", "gain_pct", "hyb_mse_diff_pct", "hyb_mse_p_runs", "hyb_mse_p_time",
                    "hyb_mse_coins_better", "hyb_mse_verdict", "hyb_qlike_diff_pct", "hyb_qlike_verdict"]].round(4).to_string(index=False))
    print(mcs.round(4).to_string(index=False))


# ============================================================================================ limits
def limits(rate: float = 0.05):
    """The forecasts used as range limits (2026-11-22; fixed before anything of it was computed for the hybrid; known
    before: the same evaluation of paper v2 for three-seed averages of other models, tables/v2_exceed.csv).

      limit      forecast of the log range plus a constant c: the (1 - rate) quantile of the forecast errors on the
                 validation block of the coin and fold (rate = 5%); nothing is fitted on the test blocks.
      measures   per run, on the 48,600 test forecasts: share of bars whose realised range exceeds the limit; the mean
                 limit in units of the range relative to that of HAR (mean of the per-coin ratios; a lower limit at the
                 same exceedance rate ties up less margin); the tick loss of the limit as a 95% quantile forecast of
                 the log range (Koenker and Bassett 1978). Mean and standard deviation over the five runs.
      models     the 13 models of the main table.
      tests      the proposed hybrid against every other model on the tick loss, with the two tests and the rule of
                 paper_v3_tests (Welch over runs; Diebold-Mariano on the tick-loss difference per bar, averaged over
                 runs and coins)."""
    from scipy import stats
    from src.paper.evaluate import diebold_mariano, holm
    from paper_v3_tests import verdict
    M, _ = build()
    per_run, per_bar = {}, {}
    for name in MAIN:
        runs, bars = [], []
        for f in M[name].values():
            d = f.reset_index()
            val, test = d[d.split == "val"], d[d.split == "test"].copy()
            c = (val.y - val.pred).groupby([val.dataset, val.fold]).quantile(1 - rate).rename("c")
            test = test.join(c, on=["dataset", "fold"])
            test["limit"] = test.pred + test.c
            u = test.y - test.limit
            test["tick"] = u * ((1 - rate) - (u < 0))
            assert len(test) == 48600
            runs.append(dict(exceed=float((u > 0).mean()), tick=float(test.tick.mean()), limit=np.exp(test.limit).groupby(test.dataset).mean()))
            bars.append(test.set_index(["time", "dataset"]).tick.sort_index())
        per_run[name], per_bar[name] = runs, sum(bars) / len(bars)
    har_limit = per_run["HAR"][0]["limit"]
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    rows = []
    for name in MAIN:
        r = per_run[name]
        tick, ex = np.array([x["tick"] for x in r]), np.array([x["exceed"] for x in r])
        lim = np.array([float((x["limit"] / har_limit).mean()) for x in r])
        row = dict(model=name, runs=len(r), exceed_mean=ex.mean(), exceed_sd=sd(ex), limit_har_mean=lim.mean(), limit_har_sd=sd(lim),
                   tick_mean=tick.mean(), tick_sd=sd(tick), tick_har=tick.mean() / per_run["HAR"][0]["tick"])
        if name != HYBRID:
            o = np.array([x["tick"] for x in per_run[HYBRID]])
            p_t = stats.ttest_ind(o, tick, equal_var=False).pvalue if len(tick) > 1 else stats.ttest_1samp(o, tick[0]).pvalue
            d = (per_bar[HYBRID] - per_bar[name]).unstack("dataset")
            series = d.mean(axis=1).sort_index().to_numpy()
            dm, p_dm = diebold_mariano(series, np.zeros(len(series)))
            assert np.isclose(series.mean(), o.mean() - tick.mean(), rtol=1e-6, atol=1e-10)
            row.update(diff_pct=100 * (o.mean() / tick.mean() - 1), p_runs=p_t, p_time=p_dm, coins_better=int((d.mean(axis=0) < 0).sum()),
                       verdict=verdict(o.mean() - tick.mean(), p_t, p_dm))
        rows.append(row)
    t = pd.DataFrame(rows)
    ok = t.p_runs.notna()
    t.loc[ok, "p_runs_holm"], t.loc[ok, "p_time_holm"] = holm(t.loc[ok, "p_runs"].to_numpy()), holm(t.loc[ok, "p_time"].to_numpy())
    t.to_csv(os.path.join(CSV, "v3_paper_limits.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.round(4).to_string(index=False))


# ============================================================================================ tables
PARAMS = {"HAR": "4", "HAR-S x": "45", LGBM: "trees", "DLinear": "1,009", "CNN, 6 convolutions": "20,585", "QCNN, 6 convolutions": "19,021",
          "PatchTST": "19,291", "iTransformer": "18,913", "Transformer": "18,945", TWIN: "20,142", NET: "19,875", TWIN_HYBRID: "", HYBRID: ""}
TEX = {"HAR": "HAR", "HAR-S x": r"HAR-S$\times$", LGBM: "LightGBM", "DLinear": "DLinear", "CNN, 6 convolutions": "CNN", "QCNN, 6 convolutions": "QCNN",
       "PatchTST": "PatchTST", "iTransformer": "iTransformer", "Transformer": "Transformer", TWIN: "H-LSTM (real twin)", NET: r"\HQ{}",
       TWIN_HYBRID: "H-LSTM + LightGBM", HYBRID: r"\textbf{\HQ{} + LightGBM}"}
GROUPS_MAIN = [("Linear models, one per coin", ["HAR", "HAR-S x"]), ("Gradient boosting, pooled", [LGBM]),
               ("Networks, pooled", ["DLinear", "CNN, 6 convolutions", "QCNN, 6 convolutions", "PatchTST", "iTransformer", "Transformer", TWIN, NET]),
               ("Hybrids: plain average of a network and LightGBM", [TWIN_HYBRID, HYBRID])]


def pm(m: float, s: float, d: int = 4) -> str:
    """mean ± sd; a positive sd that rounds to zero is printed as "< one unit of the last digit", not as zero."""
    if not np.isfinite(s):
        return f"{m:.{d}f}"
    return f"{m:.{d}f}" + (rf"\,$\pm$\,$<${10 ** -d:.{d}f}" if 0 < s < 0.5 * 10 ** -d else rf"\,$\pm$\,{s:.{d}f}")


def pv(p: float) -> str:
    return "" if not np.isfinite(p) else ("$<$0.001" if p < 0.0005 else f"{p:.3f}")


def dp(x: float, d: int = 2, bold: bool = False) -> str:
    if not np.isfinite(x):
        return ""
    v = f"{abs(x):.{d}f}"
    if float(v) == 0:
        t = v
    else:
        t = ("$-$" if x < 0 else "+") + v
    return rf"\textbf{{{t}}}" if bold else t


def write(name: str, lines: list[str]):
    os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
    with open(os.path.join(PAPER, "tables", name), "w") as f:
        f.write("% generated by experiments/paper_v3_paper.py tables\n" + "\n".join(lines) + "\n")


def tab_main():
    m = pd.read_csv(os.path.join(CSV, "v3_paper_models.csv")).set_index("model")
    t = pd.read_csv(os.path.join(CSV, "v3_tests.csv"))
    t = t[t.ours == HYBRID].set_index("other").rename(index={"H-QLSTM": NET})
    mcs = pd.read_csv(os.path.join(CSV, "v3_paper_mcs.csv"))
    mcs = mcs[mcs.loss == "mse"].set_index("model").mcs_p
    big = {"DLinear", "CNN, 6 convolutions", "QCNN, 6 convolutions"}
    L = [r"\begin{tabular}{@{}lrcccrccrrccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{MSE} & QLIKE & \multicolumn{4}{c}{Hybrid against the model, MSE} & \multicolumn{3}{c}{Hybrid against the model, QLIKE} & MCS \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(lr){6-9}\cmidrule(lr){10-12}\cmidrule(l){13-13}",
         r"Model & Param. & mean\,$\pm$\,sd & $\times$\,HAR & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & coins & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & $p$ \\",
         r"\midrule"]
    for g, (head, names) in enumerate(GROUPS_MAIN):
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
        for n in names:
            r = m.loc[n]
            if n in big:
                ql = (f"{r.qlike_mean:.1f}" if r.qlike_mean < 100 else rf"$>$10$^{{3}}$") + r"$^{\dagger}$"
            else:
                ql = pm(r.qlike_mean, r.qlike_sd, 3)
            cells = [TEX[n] + (" (proposed)" if n == HYBRID else ""), PARAMS[n], pm(r.mse_mean, r.mse_sd), f"{r.mse_har:.3f}", ql]
            if n == HYBRID:
                cells += [""] * 7
            else:
                x = t.loc[n]
                cells += [dp(x.mse_diff_pct, 1, x.mse_verdict == "better"), pv(x.mse_p_runs), pv(x.mse_p_time), f"{int(x.mse_coins_better)}/15",
                          (r"$\dagger$" if n in big else dp(x.qlike_diff_pct, 1, x.qlike_verdict == "better")), pv(x.qlike_p_runs), pv(x.qlike_p_time)]
            cells.append(pv(mcs.loc[n]) if mcs.loc[n] < 0.9995 else "1.000")
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("main.tex", L)


def tab_combos():
    c = pd.read_csv(os.path.join(CSV, "v3_paper_combos.csv"))
    L = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{MSE} & \multicolumn{3}{c}{QLIKE} \\", r"\cmidrule(lr){2-5}\cmidrule(l){6-8}",
         r"Scheme & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{time}}$ \\", r"\midrule"]
    for g, (group, rows) in enumerate(c.groupby("group", sort=False)):
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{group}}}}} \\")
        for r in rows.itertuples():
            first = r.name == HYBRID
            L.append(" & ".join([r.scheme, pm(r.mse_mean, r.mse_sd), "" if first else dp(r.avg_mse_diff_pct), "" if first else pv(r.avg_mse_p_runs),
                                 "" if first else pv(r.avg_mse_p_time), pm(r.qlike_mean, r.qlike_sd, 3), "" if first else dp(r.avg_qlike_diff_pct, 1),
                                 "" if first else pv(r.avg_qlike_p_time)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("combos.tex", L)


def tab_partners():
    p = pd.read_csv(os.path.join(CSV, "v3_paper_partners.csv"))
    lab = {NET: r"\HQ{}", TWIN: "H-LSTM", LGBM: "LightGBM", "HAR-S x": r"HAR-S$\times$"}
    name = lambda a, b: (f"Two runs of {lab.get(a, a)}" if a == b else f"{lab.get(a, a)} + {lab.get(b, b)}")
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & Error & \multicolumn{4}{c}{MSE} & QLIKE \\", r"\cmidrule(lr){3-6}\cmidrule(l){7-7}",
         r"Members of the average & corr. & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & mean \\", r"\midrule"]
    blocks = [("With LightGBM", p[(p.b == LGBM)]), ("Without LightGBM", p[(p.b != LGBM)])]
    for g, (head, rows) in enumerate(blocks):
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
        for r in rows.itertuples():
            first = (r.a, r.b) == (NET, LGBM)
            L.append(" & ".join([name(r.a, r.b) + (" (proposed)" if first else ""), f"{r.corr:.3f}", pm(r.mse_mean, r.mse_sd),
                                 "" if first else dp(r.hyb_mse_diff_pct), "" if first else pv(r.hyb_mse_p_runs), "" if first else pv(r.hyb_mse_p_time),
                                 f"{r.qlike_mean:.3f}"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("partners.tex", L)


def tab_inputs():
    x = pd.read_csv(os.path.join(CSV, "v3_inputs.csv"))
    order = ["LightGBM", "H-LSTM (real twin)", "H-QLSTM", "H-QLSTM + LightGBM"]
    head = {"LightGBM": "LightGBM", "H-LSTM (real twin)": "H-LSTM (real twin)", "H-QLSTM": r"\HQ{}", "H-QLSTM + LightGBM": r"\HQ{} + LightGBM"}
    lab = {"price, market, calendar": "Price, market, calendar (reference)", "without market": r"\quad without market", "without calendar": r"\quad without calendar",
           "+ technical": r"\quad + technical", "+ social": r"\quad + social", "+ sentiment": r"\quad + sentiment",
           "+ social, sentiment": r"\quad + social and sentiment", "+ technical, social, sentiment": r"\quad + technical, social, sentiment"}
    L = [r"\begin{tabular}{@{}l" + "cr" * 4 + r"@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{head[m]}}}" for m in order) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4)),
         "Inputs & " + " & ".join([r"MSE & $\Delta$ (\%)"] * 4) + r" \\", r"\midrule"]
    from scipy import stats
    ref = "price, market, calendar"
    for inputs in lab:
        cells = [lab[inputs]]
        for m in order:
            r = x[(x.model == m) & (x.inputs == inputs)].iloc[0]
            if inputs == ref:
                cells += [pm(r.mse_mean, r.mse_sd), ""]
                continue
            base = x[(x.model == m) & (x.inputs == ref)].iloc[0]
            # over runs: Welch's t-test from the means and standard deviations of the five runs of each input set;
            # over time: the DM test of paper_v3_inputs (forecasts averaged over the runs), Holm over the seven sets
            p_runs = stats.ttest_ind_from_stats(r.mse_mean, r.mse_sd, int(r.runs), base.mse_mean, base.mse_sd, int(base.runs), equal_var=False).pvalue
            star = "$^{*}$" if (p_runs < 0.05 and r.p_ref_holm < 0.05) else ""
            cells += [pm(r.mse_mean, r.mse_sd), dp(100 * (r.mse_mean / base.mse_mean - 1), 1) + star]
        L.append(" & ".join(cells) + r" \\")
        if inputs == "price, market, calendar":
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("inputs.tex", L)


def tab_robust():
    path = os.path.join(CSV, "v3_bars.csv")
    if not os.path.exists(path):
        print("tab_robust: tables/v3_bars.csv is not there yet")
        return
    b = pd.read_csv(path)
    lab = {"HAR": "HAR", "HAR-S x": r"HAR-S$\times$", LGBM: "LightGBM", TWIN: "H-LSTM (real twin)", "H-QLSTM": r"\HQ{}",
           TWIN_HYBRID: "H-LSTM + LightGBM", HYBRID: r"\HQ{} + LightGBM"}
    bars = {"1h": "Hourly bars", "4h": "Four-hour bars", "1d": "Daily bars"}
    L = [r"\begin{tabular}{@{}lcccrcccrcc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{MSE} & QLIKE & \multicolumn{4}{c}{Hybrid against the model, MSE} & \multicolumn{3}{c}{Hybrid against the model, QLIKE} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-8}\cmidrule(l){9-11}",
         r"Model & mean\,$\pm$\,sd & $\times$\,HAR & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & coins & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ \\",
         r"\midrule"]
    for g, freq in enumerate(bars):
        rows = b[b.bars == freq]
        if not len(rows):
            continue
        if g:
            L.append(r"\addlinespace[2pt]")
        n = int(rows.test_coin_bars.iloc[0])
        L.append(rf"\multicolumn{{11}}{{@{{}}l}}{{\emph{{{bars[freq]}}} ({n:,} test forecasts, {int(rows.runs.max())} runs)}} \\")
        for r in rows.itertuples():
            cells = [lab[r.model], pm(r.mse_mean, r.mse_sd), f"{r.mse_har:.3f}", pm(r.qlike_mean, r.qlike_sd, 3)]
            if r.model == HYBRID:
                cells += [""] * 7
            else:
                cells += [dp(r.mse_diff_pct, 1, r.mse_verdict == "better"), pv(r.mse_p_runs), pv(r.mse_p_time), f"{int(r.mse_coins_better)}/15",
                          dp(r.qlike_diff_pct, 1, r.qlike_verdict == "better"), pv(r.qlike_p_runs), pv(r.qlike_p_time)]
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("robust.tex", L)


def tables():
    for fn in (tab_main, tab_combos, tab_partners, tab_inputs, tab_robust):
        fn()
    for f in sorted(os.listdir(os.path.join(PAPER, "tables"))):
        print(f)



# ============================================================================================ figures
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b6b6b"      # Okabe-Ito, as in the figures of paper v2
SHORT = {"HAR": "HAR", "HAR-S x": "HAR-S×", LGBM: "LightGBM", "DLinear": "DLinear", "CNN, 6 convolutions": "CNN", "QCNN, 6 convolutions": "QCNN",
         "PatchTST": "PatchTST", "iTransformer": "iTransformer", "Transformer": "Transformer", TWIN: "H-LSTM (real twin)", NET: "H-QLSTM",
         TWIN_HYBRID: "H-LSTM + LightGBM", HYBRID: "H-QLSTM + LightGBM"}
FAMILY = {"HAR": (GREY, "s"), "HAR-S x": (GREY, "s"), LGBM: (ORANGE, "D"), TWIN_HYBRID: (RED, "o"), HYBRID: (RED, "o")}      # networks: blue circles


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
    return plt


def fig_data(coin: str = "btc", freq: str = "4h", zoom_days: int = 21):
    """The target with the blocks of the four folds, and the three input quaternions of the proposed network."""
    import matplotlib.dates as mdates
    from src.paper.data import GROUPS, make_dataset, walk_forward_folds
    plt = _style()
    colours = [BLUE, ORANGE, GREEN, RED]
    panels = [("price_r", "Price (with the log range)", ["log range", "high", "low", "close"]),
              ("market", "Market", ["volume", "trades", "taker imbalance", r"$\Delta$ volume"]),
              ("calendar", "Calendar of the next bar", ["time of day, sin", "time of day, cos", "week, sin", "week, cos"])]
    d = make_dataset(coin, freq, root="data")
    f, folds = d.frame, walk_forward_folds(d.frame.index, n_folds=4, test_days=135, val_days=180)
    last = folds[-1]
    train = f.iloc[last["train"]]
    fig = plt.figure(figsize=(7.2, 3.3))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.95, wspace=0.2)
    ax = fig.add_subplot(grid[0, :])
    ax.plot(f.index, f["log_range"], color="0.62", lw=0.35, rasterized=True)
    ax.plot(f.index, f["log_range"].rolling(42).mean(), color="0.15", lw=0.9)
    for k, fold in enumerate(folds):
        ax.axvspan(fold["test_start"], fold["test_end"], color=BLUE, alpha=0.16, lw=0)
        ax.axvline(fold["test_start"], color="white", lw=1.2)
        mid = fold["test_start"] + (fold["test_end"] - fold["test_start"]) / 2
        ax.text(mid, 1.02, f"test {k + 1}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    first = folds[0]
    val_start = first["test_start"] - pd.Timedelta(days=180)
    ax.axvspan(val_start, first["test_start"], color="0.55", alpha=0.16, lw=0)
    ax.text(val_start + (first["test_start"] - val_start) / 2, 1.02, "valid. 1", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    ax.text(f.index[0] + (val_start - f.index[0]) / 2, 1.02, "training block of fold 1 (expands by one test block per fold)",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("log range")
    ax.set_xlim(f.index[0], f.index[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"(a) {coin.upper()}, four-hour bars: log range (grey) and its one-week mean (black)", loc="left", pad=14)
    test = f.iloc[last["test"]]
    centre = test["log_range"].idxmax()
    half = pd.Timedelta(days=zoom_days / 2)
    lo, hi = max(centre - half, test.index[0]), min(centre + half, test.index[-1])
    z = f.loc[lo:hi]
    for n, (group, title, names) in enumerate(panels):
        ax = fig.add_subplot(grid[1, n])
        for col, name, colour in zip(GROUPS[group], names, colours):
            sd = train[col].std()
            ax.plot(z.index, (z[col] - train[col].mean()) / (sd if sd > 0 else 1.0), color=colour, lw=0.8, label=name)
        ax.axhline(0, color="0.8", lw=0.4, zorder=0)
        ax.set_title(f"({'bcd'[n]}) {title}", loc="left", pad=21)
        ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.99), frameon=False, handlelength=1.4, columnspacing=1.0,
                  handletextpad=0.4, borderaxespad=0.0, labelspacing=0.25)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        if n == 0:
            ax.set_ylabel("z-score")
    fig.savefig(os.path.join(PAPER, "figures", "fig_data.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"fig_data: zoom {lo:%Y-%m-%d} to {hi:%Y-%m-%d}, most volatile test bar {centre:%Y-%m-%d %H:%M}")


def fig_main():
    """(a) MSE of the five runs of every model; (b) the hybrid against the closest models; (c) MSE per coin."""
    plt = _style()
    import paper_v2_round2 as R
    from paper_v3_tests import bar_losses
    m = pd.read_csv(os.path.join(CSV, "v3_paper_models.csv")).set_index("model")
    t = pd.read_csv(os.path.join(CSV, "v3_tests.csv"))
    t = t[t.ours == HYBRID].set_index("other").rename(index={"H-QLSTM": NET})
    coins = pd.read_csv(os.path.join(CSV, "v3_paper_coins.csv"), index_col=0)
    M, _ = build()
    caps = R.train_caps()
    single = {n: [bar_losses(f, caps).mse.mean() for f in M[n].values()] for n in MAIN}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.25, 0.95, 1.1], wspace=0.07))
    order = MAIN[::-1]
    ax = axes[0]
    for y, n in enumerate(order):
        colour, marker = FAMILY.get(n, (BLUE, "o"))
        ax.plot(single[n], [y] * len(single[n]), ls="", marker="|", ms=5, color="0.55", mew=0.6, zorder=2)
        ax.plot(np.mean(single[n]), y, ls="", marker=marker, ms=4.2, color=colour, mec="white", mew=0.5, zorder=3)
        ax.text(np.mean(single[n]) + 0.0016 + (np.max(single[n]) - np.mean(single[n])), y, f"{m.loc[n, 'mse_mean']:.4f}", va="center", fontsize=6.3, color="0.25")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[n] for n in order])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(0.1805, 0.236)
    ax.set_xlabel("MSE of the log range")
    ax.set_title("(a) Five runs of every model", loc="left")
    ax.grid(axis="x", color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    ax = axes[1]
    close = [LGBM, "PatchTST", "iTransformer", "Transformer", TWIN, NET, TWIN_HYBRID]
    for y, n in enumerate(order):
        if n not in close:
            if n != HYBRID:
                ax.annotate(f"{t.loc[n, 'mse_diff_pct']:.1f}%", (-6.35, y), ha="left", va="center", fontsize=6.3, color="0.35")
                ax.annotate("", xy=(-6.6, y), xytext=(-5.2, y) if False else (-6.38, y), arrowprops=dict(arrowstyle="-", lw=0))
            continue
        r = t.loc[n]
        colour, marker = FAMILY.get(n, (BLUE, "o"))
        ax.plot([r.mse_diff_lo, r.mse_diff_hi], [y, y], color=colour, lw=1.0, solid_capstyle="butt")
        ax.plot(r.mse_diff_pct, y, ls="", marker=marker, ms=4.2, color=colour, mec="white", mew=0.5)
    ax.axvline(0, color="0.3", lw=0.6)
    ax.set_ylim(axes[0].get_ylim())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([])
    ax.set_xlim(-6.6, 0.9)
    ax.set_xlabel("MSE of the hybrid against the model (%)")
    ax.set_title("(b) Difference, 90% interval", loc="left")
    ax.grid(axis="x", color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    ax = axes[2]
    rel = coins.div(coins["HAR"], axis=0).sort_values(HYBRID, ascending=False)
    for col, colour, marker, label in ((LGBM, ORANGE, "D", "LightGBM"), (NET, BLUE, "o", "H-QLSTM"), (HYBRID, RED, "o", "hybrid")):
        ax.plot(rel[col], range(len(rel)), ls="", marker=marker, ms=3.6, color=colour, mec="white", mew=0.4, label=label)
    for y, (lo, hi) in enumerate(zip(rel[[LGBM, NET, HYBRID]].min(axis=1), rel[[LGBM, NET, HYBRID]].max(axis=1))):
        ax.plot([lo, hi], [y, y], color="0.8", lw=0.6, zorder=0)
    ax.set_yticks(range(len(rel)))
    ax.set_yticklabels([c.split("_")[0].upper() for c in rel.index])
    ax.yaxis.tick_right()
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("MSE relative to HAR")
    ax.set_title("(c) Per coin", loc="left")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    ax.grid(axis="x", color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(PAPER, "figures", "fig_main.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_combination():
    """(a) correlation of the forecast errors; (b) MSE against the weight of the network; (c) cumulative gain over the test period."""
    plt = _style()
    import matplotlib.dates as mdates
    corr = pd.read_csv(os.path.join(CSV, "v3_paper_corr.csv")).pivot(index="a", columns="b", values="corr")
    names = ["HAR", "HAR-S x", LGBM, "PatchTST", "iTransformer", "Transformer", TWIN, NET]
    corr = corr.loc[names, names]
    w = pd.read_csv(os.path.join(CSV, "v3_paper_weight.csv"))
    fitted = pd.read_csv(os.path.join(CSV, "v3_paper_weights.csv"))
    tm = pd.read_csv(os.path.join(CSV, "v3_paper_time.csv"), index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw=dict(width_ratios=[1.05, 0.9, 1.15], wspace=0.42))
    ax = axes[0]
    v = corr.to_numpy()
    shown = np.where(np.isnan(v), 1.0, v)
    ax.imshow(shown, cmap="Blues", vmin=0.88, vmax=1.0)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(v[i, j]):
                ax.text(j, i, f"{v[i, j]:.3f}"[1:], ha="center", va="center", fontsize=5.0, color="white" if v[i, j] > 0.962 else "0.15")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=5.6, color="white")
    lab = [SHORT[n].replace(" (real twin)", "") for n in names]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(lab, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(lab)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("(a) Correlation of forecast errors", loc="left")
    ax = axes[1]
    for split, ls, label in (("test", "-", "test blocks"), ("val", "--", "validation blocks")):
        g = w[(w.network == NET) & (w.split == split)]
        ax.plot(g.w, g.mse, color=BLUE, ls=ls, lw=1.1, label=label)
    f = fitted[fitted.scheme == "W2"].weight_on_network
    ax.axvspan(f.min(), f.max(), color=ORANGE, alpha=0.22, lw=0)
    ax.axvline(0.5, color=RED, lw=0.8)
    ymax = w[(w.network == NET)].mse.max()
    ax.text(0.49, ymax, "plain\naverage", ha="right", va="top", fontsize=6.3, color=RED)
    ax.text(f.max() + 0.02, ymax - 0.0004, "fitted\nweights", ha="left", va="top", fontsize=6.3, color="#9a6a00")
    ax.set_xlabel("weight of the H-QLSTM")
    ax.set_ylabel("MSE")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0\nLightGBM", "0.25", "0.5", "0.75", "1\nnetwork"])
    ax.set_title("(b) MSE against the weight", loc="left")
    ax = axes[2]
    for col, colour, ls, label in (("hybrid_lgbm_mse", ORANGE, "-", "LightGBM"), ("hybrid_net_mse", BLUE, "-", "H-QLSTM"),
                                   ("hybrid_transformer_mse", GREEN, "-", "Transformer"), ("hybrid_twinhybrid_mse", RED, "--", "H-LSTM + LightGBM")):
        ax.plot(tm.index, tm[col].cumsum(), color=colour, ls=ls, lw=1.0, label=label)
    for start in tm.groupby("fold").apply(lambda g: g.index[0])[1:]:
        ax.axvline(start, color="0.75", lw=0.5, ls=":")
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel("cumulative difference in squared error")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="lower left", frameon=False, borderaxespad=0.1, title="hybrid minus", title_fontsize=6.5)
    ax.set_title("(c) Hybrid minus model, over time", loc="left")
    fig.savefig(os.path.join(PAPER, "figures", "fig_combination.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_controls():
    """Design controls of the network (three-seed averages; ten where noted): each control against the proposed network,
    and the quaternion network against its real twin in every control."""
    plt = _style()
    g = pd.read_csv(os.path.join(CSV, "v3_grid.csv"))
    sm = pd.read_csv(os.path.join(CSV, "v3_small.csv")).set_index("pair")
    se = pd.read_csv(os.path.join(CSV, "v3_grid_seeds.csv")).set_index("pair")
    bars = pd.read_csv(os.path.join(CSV, "v3_grid_bars.csv"))
    rows = [("Proposed network, 20,000 parameters (10 seeds)", None, (se.loc["Candidate A / real twin", "ratio_all"], se.loc["Candidate A / real twin", "ci_lo"], se.loc["Candidate A / real twin", "ci_hi"]))]
    label = {"without the range in the price quaternion": "Inputs: price without the range", "without the calendar group": "Inputs: without calendar",
             "without the market group": "Inputs: without market", "with social and sentiment groups": "Inputs: + social and sentiment",
             "additive term instead of a group": "Calendar as an additive term", "concatenation": "Fusion: concatenation", "dense": "Fusion: dense layer",
             "two layers": "Two recurrent layers", "flat QLSTM / LSTM": "One encoder for all groups", "5,000 parameters": "5,000 parameters",
             "80,000 parameters": "80,000 parameters", "3e-3": "Learning rate 3e-3", "3e-4": "Learning rate 3e-4", "one model per coin": "One model per coin"}
    for r in g.itertuples():
        if r.control in label:
            rows.append((label[r.control], (r.vs_A, r.vs_A_lo, r.vs_A_hi), (r.twin_ratio, r.twin_ci_lo, r.twin_ci_hi)))
        if r.control == "flat QLSTM / LSTM":
            for n in ("500", "1,000", "2,000"):
                a, b = sm.loc[f"H-QLSTM, {n} / 20,000 parameters"], sm.loc[f"{n} parameters: H-QLSTM / real twin"]
                rows.append((f"{n} parameters (10 seeds)", (a.ratio_all, a.ci_lo, a.ci_hi), (b.ratio_all, b.ci_lo, b.ci_hi)))
    for freq, name in (("1h", "Hourly bars"), ("1d", "Daily bars")):
        b = bars[(bars.bars == freq) & (bars.model == "Candidate A, real twin")].iloc[0]
        rows.append((name, None, (b.A_over_model, b.ci_lo, b.ci_hi)))
    rows = rows[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.06))
    for ax, k, colour, title, xlim in ((axes[0], 1, BLUE, "(a) Control against the proposed network", (0.988, 1.066)),
                                       (axes[1], 2, RED, "(b) Quaternion network against its real twin", (0.988, 1.027))):
        for y, row in enumerate(rows):
            if row[k] is None:
                continue
            v, lo, hi = row[k]
            ax.plot([lo, hi], [y, y], color=colour, lw=1.0, solid_capstyle="butt")
            ax.plot(v, y, ls="", marker="o", ms=3.8, color=colour, mec="white", mew=0.5)
        ax.axvline(1, color="0.3", lw=0.6)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_yticks(range(len(rows)))
        ax.set_title(title, loc="left")
        ax.set_xlabel("ratio of MSEs, 90% interval")
        ax.grid(axis="x", color="0.9", lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[1].set_yticklabels([])
    fig.savefig(os.path.join(PAPER, "figures", "fig_controls.pdf"), bbox_inches="tight")
    plt.close(fig)


def figs():
    for fn in (fig_data, fig_main, fig_combination, fig_controls):
        fn()
        print(fn.__name__)



if __name__ == "__main__":
    fns = {"stats": stats, "tables": tables, "figs": figs, "limits": limits}
    for a in sys.argv[1:] or ["stats"]:
        fns[a]()
