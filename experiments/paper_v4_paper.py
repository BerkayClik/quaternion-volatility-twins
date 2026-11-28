"""
Statistics, tables and figures of manuscript v4 (docs/paper/manuscript_v4: the dual-view hybrid), computed from the
saved forecasts. No model is trained here.

Plan (2026-11-23, fixed before any number below is computed; committed before the first run of this file).

  framing    The proposed system is the dual-view hybrid: the plain average of a hierarchical recurrent network
             that reads the window of bars and the lagged LightGBM-S that reads summaries of the same bars. The
             network has two versions of equal size, with real LSTM layers (H-LSTM, `hl_gattn`) and with quaternion
             LSTM layers (H-QLSTM, `hq_gattn`). The paper presents the version with real layers as the default and
             reports the quaternion version next to it in every table.
  why        The choice of the default was made with these results known (tables/v3_paper_models.csv, v3_bars.csv,
             v3_paper_mcs.csv, v3_paper_limits.csv, v3_hybrid.csv): on 4-hour bars the hybrid with real layers has
             an MSE of 0.1825 +- 0.0002 against 0.1828 +- 0.0003 (difference 0.14%, not significant), it is 0.25%
             better on hourly bars (both tests), equal on daily bars, the only model in the 90% model confidence set,
             and equal in the tick loss of the range limits. The system was developed with the quaternion version;
             the real version was its control. The manuscript states this order.
  not known  Not computed before this plan: the tests of the hybrid with real layers against the single models (MSE,
             QLIKE, tick loss of the range limits, on any bar length); stacking and the three-model schemes with
             the real network; pairs of single models against the hybrid with real layers; the hybrid with real
             layers on the other input sets.

  tests      tables/v4_tests.csv: each of the two hybrids against every other model of the main table, MSE and
             QLIKE, with the two tests and the rule of paper_v3_tests (Welch over the five runs, one-sample against
             the two HAR models; Diebold-Mariano over time on the loss difference per bar averaged over runs and
             coins; better or worse only if both p < 0.05). Holm per hybrid, loss and test over the 12 comparisons.
             The rows of the quaternion version must equal tables/v3_tests.csv.
  combos     tables/v4_combos.csv: for each version the schemes that need no further training (plain, weighted and
             stacked averages with LightGBM, with HAR or HAR-S x, and of three models) and the trained cascades that
             exist (LightGBM then network for both versions; the cascades that start from HAR and the three-stage
             cascades for the quaternion version only), against the plain average of that version and against
             LightGBM. The rows of the quaternion version must equal tables/v3_paper_combos.csv.
  partners   tables/v4_partners.csv: plain averages of two single models against the hybrid with real layers.
  limits     tables/v4_limits.csv: the 95% range limits of paper_v3_paper.limits (same constant, measures and
             models), each of the two hybrids against every other model on the tick loss. The statistics per model
             must equal tables/v3_paper_limits.csv. tables/v4_limits_example.csv: the limits of HAR and of the
             hybrid with real layers (run 0) for BTC in the last test block, for a figure.
  bars       tables/v4_bars.csv: paper_v3_bars with each of the two hybrids against the other models, hourly,
             4-hour and daily bars. The result of the 4-hour bars counts as confirmed for a version on a bar length
             only if that version has a lower MSE than LightGBM and than its own network by the rule.
  inputs     tables/v4_inputs.csv: the hybrid with real layers on the eight input sets of paper_v3_inputs (run k of
             the real network with run k of the LightGBM model of that input set): MSE and QLIKE over five runs, and
             each set against the reference set with Welch's test over the runs and the Diebold-Mariano test of
             paper_v3_inputs (forecasts averaged over the runs; Holm over the seven sets).
  time       tables/v4_time.csv: loss difference per test bar (mean over coins and runs) of the hybrid with real
             layers against LightGBM, its network, the Transformer, HAR-S x and the quaternion version.

    .venv/bin/python -W ignore experiments/paper_v4_paper.py [stats|limits|bars|inputs|extras|tables|figs]
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_paper as P3  # noqa: E402
import paper_v3_runs as RU  # noqa: E402
from paper_v3_paper import BLUE, GREEN, GREY, LGBM, ORANGE, RED, compare, dp, pair_name, pm, pv  # noqa: E402

CSV, RUNS = RU.OUT, RU.RUNS
PAPER, PAPER_V3 = "docs/paper/manuscript_v4", "docs/paper/manuscript_v3"
Q_NET, R_NET, Q_HYB, R_HYB = P3.NET, P3.TWIN, P3.HYBRID, P3.TWIN_HYBRID
MAIN = ["HAR", "HAR-S x", LGBM, "DLinear", "CNN, 6 convolutions", "QCNN, 6 convolutions", "PatchTST", "iTransformer", "Transformer",
        Q_NET, R_NET, Q_HYB, R_HYB]
HYBRIDS = [R_HYB, Q_HYB]
CASCADE_R, CASCADE_Q = "Cascade: LightGBM, then H-LSTM", "Cascade: LightGBM, then H-QLSTM"
# version -> (group, label, name in `build`)
COMBOS = {
    R_HYB: [
        ("H-LSTM and LightGBM", "Plain average (the hybrid)", R_HYB),
        ("H-LSTM and LightGBM", "Weighted average", "W2 twin"),
        ("H-LSTM and LightGBM", "Linear stacking", "S2 real"),
        ("H-LSTM and LightGBM", "Cascade: LightGBM, then H-LSTM", CASCADE_R),
        ("H-LSTM and HAR", "Plain average with HAR", "A2H real"),
        ("H-LSTM and HAR", "Plain average with HAR-S$\\times$", "A2HS real"),
        ("HAR, LightGBM and H-LSTM", "Plain average", "A3 real"),
        ("HAR, LightGBM and H-LSTM", "Weighted average", "W3 real"),
        ("HAR, LightGBM and H-LSTM", "Linear stacking", "S3 real"),
        ("HAR-S$\\times$, LightGBM and H-LSTM", "Plain average", "A3S real"),
        ("HAR-S$\\times$, LightGBM and H-LSTM", "Weighted average", "W3S real"),
        ("HAR-S$\\times$, LightGBM and H-LSTM", "Linear stacking", "S3S real"),
    ],
    Q_HYB: P3.COMBOS,
}
PARTNERS = [(R_NET, LGBM), (Q_NET, LGBM), ("Transformer", LGBM), ("PatchTST", LGBM), ("iTransformer", LGBM), ("HAR-S x", LGBM), (LGBM, LGBM),
            (R_NET, R_NET), (Q_NET, Q_NET), (R_NET, Q_NET), (R_NET, "Transformer"), (R_NET, "PatchTST"), (R_NET, "HAR-S x")]
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def build() -> tuple[dict, dict]:
    """The models and combinations of paper_v3_paper.build, and the schemes with the real network that it does not hold."""
    import paper_v3_triple as TR
    M, weights = P3.build()
    T, G, har, harsx = M[R_NET], M[LGBM], M["HAR"][0], M["HAR-S x"][0]
    mix = lambda a, b: a.assign(pred=(a.pred + b.reindex(a.index).pred) / 2)
    M["S2 real"] = {k: RU.stack(T[k], G[k].reindex(T[k].index)) for k in RUNS}
    M["A2H real"], M["A2HS real"] = {k: mix(T[k], har) for k in RUNS}, {k: mix(T[k], harsx) for k in RUNS}

    def wide(k: int, third: pd.DataFrame) -> pd.DataFrame:
        a = T[k]
        x = pd.DataFrame({"y": a.y, "h": third.reindex(a.index).pred, "g": G[k].reindex(a.index).pred, "n": a.pred})
        assert x.notna().all().all()
        return x

    for tag, third in (("", har), ("S", harsx)):
        M[f"A3{tag} real"] = {k: (lambda x: x[["y"]].assign(pred=x[["h", "g", "n"]].mean(1)))(wide(k, third)) for k in RUNS}
        M[f"W3{tag} real"] = {k: TR.blend_w(wide(k, third))[0] for k in RUNS}
        M[f"S3{tag} real"] = {k: TR.stack(wide(k, third))[0] for k in RUNS}
    one = lambda m, k: M[m][k if len(M[m]) > 1 else 0]
    for a, b in PARTNERS:                      # two runs of one model: run k with run k + 1
        if pair_name(a, b) not in M:
            M[pair_name(a, b)] = {k: mix(one(a, k), one(b, (k + 1) % len(RUNS) if a == b else k)) for k in RUNS}
    assert all(np.allclose(M[pair_name(R_NET, LGBM)][k].pred, M[R_HYB][k].pred) for k in RUNS)
    return M, weights


def _losses(M: dict, caps: pd.Series) -> tuple[dict, dict]:
    from paper_v3_tests import bar_losses
    per_run, per_bar = {}, {}
    for name, by_run in M.items():
        bars = [bar_losses(f, caps) for f in by_run.values()]
        per_run[name] = pd.DataFrame([b.mean() for b in bars])
        per_bar[name] = sum(bars) / len(bars)
    return per_run, per_bar


def _holm(t: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    from src.paper.evaluate import holm
    for _, g in t.groupby(by, sort=False):
        for col in cols:
            ok = g.index[g[col].notna()]
            t.loc[ok, col + "_holm"] = holm(t.loc[ok, col].to_numpy())
    return t


P_COLS = ["mse_p_runs", "mse_p_time", "qlike_p_runs", "qlike_p_time"]


def stats():
    import paper_v2_round2 as R
    M, _ = build()
    per_run, per_bar = _losses(M, R.train_caps())

    # tests: each hybrid against every other model of the main table
    rows = [dict(ours=ours, other=other, mse_ours=per_run[ours].mse.mean(), mse_other=per_run[other].mse.mean(),
                 qlike_ours=per_run[ours].qlike.mean(), qlike_other=per_run[other].qlike.mean(), **compare(per_run, per_bar, ours, other))
            for ours in HYBRIDS for other in MAIN if other != ours]
    tests = _holm(pd.DataFrame(rows), ["ours"], P_COLS)
    tests.to_csv(os.path.join(CSV, "v4_tests.csv"), index=False)
    old = pd.read_csv(os.path.join(CSV, "v3_tests.csv"))
    old = old[old.ours == Q_HYB].set_index("other")
    new = tests[tests.ours == Q_HYB].set_index("other")
    for other in old.index:
        for col in ("mse_diff_pct", "mse_p_runs", "mse_p_time", "qlike_diff_pct", "qlike_p_runs", "qlike_p_time"):
            assert np.isclose(old.loc[other, col], new.loc[other, col], rtol=1e-6, atol=1e-12), (other, col)

    # combos: the schemes of each version against its plain average and against LightGBM
    rows = []
    for version, schemes in COMBOS.items():
        for group, label, name in schemes:
            p = per_run[name]
            r = dict(version=version, group=group, scheme=label, name=name, mse_mean=p.mse.mean(), mse_sd=sd(p.mse), qlike_mean=p.qlike.mean(), qlike_sd=sd(p.qlike))
            if name != version:
                r.update({"avg_" + k: v for k, v in compare(per_run, per_bar, name, version).items()})
            r.update({"lgbm_" + k: v for k, v in compare(per_run, per_bar, name, LGBM).items()})
            rows.append(r)
    combos = pd.DataFrame(rows)
    combos.to_csv(os.path.join(CSV, "v4_combos.csv"), index=False)
    old = pd.read_csv(os.path.join(CSV, "v3_paper_combos.csv")).set_index("name")
    new = combos[combos.version == Q_HYB].set_index("name")
    assert np.allclose(old.mse_mean, new.loc[old.index, "mse_mean"], rtol=1e-9)
    assert np.allclose(old.avg_mse_p_time.dropna(), new.loc[old.index, "avg_mse_p_time"].dropna(), rtol=1e-6, atol=1e-12)

    # partners: plain average of two single models against the hybrid with real layers
    rows = []
    te = lambda f: f.index.get_level_values("split") == "test"
    for a, b in PARTNERS:
        name = pair_name(a, b)
        fa = lambda k: M[a][k if len(M[a]) > 1 else 0]
        fb = lambda k: M[b][((k + 1) % len(RUNS) if a == b else k) if len(M[b]) > 1 else 0]
        cc = [float(np.corrcoef((fa(k).pred - fa(k).y)[te(fa(k))], (fb(k).reindex(fa(k).index).pred - fa(k).y)[te(fa(k))])[0, 1]) for k in RUNS]
        best = min(per_run[a].mse.mean(), per_run[b].mse.mean())
        r = dict(a=a, b=b, mse_mean=per_run[name].mse.mean(), mse_sd=sd(per_run[name].mse), qlike_mean=per_run[name].qlike.mean(),
                 qlike_sd=sd(per_run[name].qlike), mse_a=per_run[a].mse.mean(), mse_b=per_run[b].mse.mean(), corr=np.mean(cc),
                 gain_pct=100 * (per_run[name].mse.mean() / best - 1))
        if (a, b) != (R_NET, LGBM):
            r.update({"hyb_" + k: v for k, v in compare(per_run, per_bar, name, R_HYB).items()})
        rows.append(r)
    partners = pd.DataFrame(rows)
    partners.to_csv(os.path.join(CSV, "v4_partners.csv"), index=False)

    # time: loss difference per test bar (mean over coins and runs)
    fold_of = M[R_NET][0][M[R_NET][0].index.get_level_values("split") == "test"].reset_index().groupby("time").fold.first()
    t = pd.DataFrame({"fold": fold_of})
    for col, (a, b) in {"hybrid_lgbm": (R_HYB, LGBM), "hybrid_net": (R_HYB, R_NET), "hybrid_harsx": (R_HYB, "HAR-S x"),
                        "hybrid_transformer": (R_HYB, "Transformer"), "hybrid_quaternion": (R_HYB, Q_HYB)}.items():
        for loss in ("mse", "qlike"):
            t[f"{col}_{loss}"] = (per_bar[a][loss] - per_bar[b][loss]).unstack("dataset").mean(axis=1)
    t.to_csv(os.path.join(CSV, "v4_time.csv"))

    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200)
    print(tests[["ours", "other", "mse_diff_pct", "mse_diff_lo", "mse_diff_hi", "mse_p_runs", "mse_p_time", "mse_coins_better", "mse_verdict",
                 "mse_p_runs_holm", "mse_p_time_holm", "qlike_diff_pct", "qlike_p_runs", "qlike_p_time", "qlike_verdict"]].round(4).to_string(index=False))
    print(combos[["version", "scheme", "mse_mean", "mse_sd", "avg_mse_diff_pct", "avg_mse_p_runs", "avg_mse_p_time", "avg_mse_verdict", "qlike_mean",
                  "avg_qlike_diff_pct", "avg_qlike_p_time", "avg_qlike_verdict", "lgbm_mse_diff_pct", "lgbm_mse_verdict"]].round(4).to_string(index=False))
    print(partners[["a", "b", "mse_mean", "mse_sd", "qlike_mean", "corr", "gain_pct", "hyb_mse_diff_pct", "hyb_mse_p_runs", "hyb_mse_p_time",
                    "hyb_mse_coins_better", "hyb_mse_verdict", "hyb_qlike_diff_pct", "hyb_qlike_verdict"]].round(4).to_string(index=False))


# ============================================================================================ limits
def limits(rate: float = 0.05):
    """The 95% range limits of paper_v3_paper.limits with each of the two hybrids against every other model."""
    from scipy import stats as st
    from src.paper.evaluate import diebold_mariano
    from paper_v3_tests import verdict
    M, _ = build()
    per_run, per_bar, frames = {}, {}, {}
    for name in MAIN:
        runs, bars = [], []
        for k, f in M[name].items():
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
            if k == 0:
                frames[name] = test[["time", "dataset", "fold", "y", "pred", "limit"]]
        per_run[name], per_bar[name] = runs, sum(bars) / len(bars)
    har_limit = per_run["HAR"][0]["limit"]
    rows = []
    for name in MAIN:
        r = per_run[name]
        tick, ex = np.array([x["tick"] for x in r]), np.array([x["exceed"] for x in r])
        lim = np.array([float((x["limit"] / har_limit).mean()) for x in r])
        base = dict(model=name, runs=len(r), exceed_mean=ex.mean(), exceed_sd=sd(ex), limit_har_mean=lim.mean(), limit_har_sd=sd(lim),
                    tick_mean=tick.mean(), tick_sd=sd(tick), tick_har=tick.mean() / per_run["HAR"][0]["tick"])
        for ours in HYBRIDS:
            row = dict(ours=ours, **base)
            if name != ours:
                o = np.array([x["tick"] for x in per_run[ours]])
                p_t = st.ttest_ind(o, tick, equal_var=False).pvalue if len(tick) > 1 else st.ttest_1samp(o, tick[0]).pvalue
                d = (per_bar[ours] - per_bar[name]).unstack("dataset")
                series = d.mean(axis=1).sort_index().to_numpy()
                dm, p_dm = diebold_mariano(series, np.zeros(len(series)))
                assert np.isclose(series.mean(), o.mean() - tick.mean(), rtol=1e-6, atol=1e-10)
                row.update(diff_pct=100 * (o.mean() / tick.mean() - 1), p_runs=p_t, p_time=p_dm, coins_better=int((d.mean(axis=0) < 0).sum()),
                           verdict=verdict(o.mean() - tick.mean(), p_t, p_dm))
            rows.append(row)
    t = _holm(pd.DataFrame(rows), ["ours"], ["p_runs", "p_time"])
    t.to_csv(os.path.join(CSV, "v4_limits.csv"), index=False)
    old = pd.read_csv(os.path.join(CSV, "v3_paper_limits.csv")).set_index("model")
    new = t[t.ours == Q_HYB].set_index("model")
    for col in ("exceed_mean", "limit_har_mean", "tick_mean", "p_time"):
        assert np.allclose(old[col].dropna(), new.loc[old.index, col].dropna(), rtol=1e-6, atol=1e-12), col
    ex = frames["HAR"][frames["HAR"].dataset.str.startswith("btc")].set_index("time")[["fold", "y", "limit"]].rename(columns={"limit": "limit_har"})
    hy = frames[R_HYB][frames[R_HYB].dataset.str.startswith("btc")].set_index("time")
    ex["pred_hybrid"], ex["limit_hybrid"] = hy.pred, hy.limit
    ex[ex.fold == ex.fold.max()].to_csv(os.path.join(CSV, "v4_limits_example.csv"))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 100)
    print(t.round(4).to_string(index=False))


# ============================================================================================ bars
def bars():
    """paper_v3_bars with each of the two hybrids against the other models."""
    import paper_v2_round2 as R
    import paper_v3_bars as B
    rows = []
    for freq, sources in B.SOURCES.items():
        caps = R.train_caps(freq)
        runs = {name: RU.forecasts(key, pats) for name, (key, pats) in sources.items()}
        mix = lambda a, b: a.assign(pred=(a.pred + b.reindex(a.index).pred) / 2)
        models = {m: {0: runs[m][0]} for m in ("HAR", "HAR-S x")}
        models.update({m: {k: runs[m][k] for k in RUNS} for m in (LGBM, R_NET, Q_NET)})
        models[R_HYB] = {k: mix(runs[R_NET][k], runs[LGBM][k]) for k in RUNS}
        models[Q_HYB] = {k: mix(runs[Q_NET][k], runs[LGBM][k]) for k in RUNS}
        per_run, per_bar = {}, {}
        for name, by_run in models.items():
            b = [B.bar_losses(f, caps) for f in by_run.values()]
            per_run[name], per_bar[name] = pd.DataFrame([x.mean() for x in b]), sum(b) / len(b)
        n = len(per_bar[R_HYB])
        har = per_run["HAR"].mean()
        for ours in HYBRIDS:
            for name in B.ORDER:
                p = per_run[name]
                r = dict(ours=ours, bars=freq, model=name, runs=len(p), test_coin_bars=n, mse_mean=p.mse.mean(), mse_sd=sd(p.mse), mse_har=p.mse.mean() / har.mse,
                         qlike_mean=p.qlike.mean(), qlike_sd=sd(p.qlike), qlike_har=p.qlike.mean() / har.qlike)
                if name != ours:
                    r.update(compare(per_run, per_bar, ours, name))
                rows.append(r)
    t = _holm(pd.DataFrame(rows), ["ours", "bars"], P_COLS)
    t.to_csv(os.path.join(CSV, "v4_bars.csv"), index=False)
    old = pd.read_csv(os.path.join(CSV, "v3_bars.csv")).set_index(["bars", "model"])
    new = t[t.ours == Q_HYB].set_index(["bars", "model"])
    for col in ("mse_mean", "qlike_mean", "mse_p_runs", "mse_p_time"):
        assert np.allclose(old[col].dropna(), new.loc[old.index, col].dropna(), rtol=1e-6, atol=1e-12), col
    pd.set_option("display.width", 280, "display.max_columns", 40, "display.max_rows", 100)
    print(t[["ours", "bars", "model", "mse_mean", "mse_sd", "mse_har", "mse_diff_pct", "mse_p_runs", "mse_p_time", "mse_coins_better", "mse_verdict",
             "mse_p_time_holm", "qlike_mean", "qlike_sd", "qlike_diff_pct", "qlike_p_runs", "qlike_p_time", "qlike_verdict"]].round(4).to_string(index=False))


# ============================================================================================ inputs
def inputs():
    """The hybrid with real layers on the input sets of paper_v3_inputs."""
    import paper_v2_round2 as R
    import paper_v3_inputs as I
    from scipy import stats as st
    caps = R.train_caps()
    har = RU.forecasts(*RU.FIXED["HAR"])[0]
    mse_har, _ = RU.losses(har, caps)
    rows, ens, per = [], {}, {}
    for name, gs, key in I.SETS:
        T = RU.forecasts((gs, "hl_gattn-pool"), I.NET_RUNS[gs])
        G = RU.forecasts(key, I.LGBM_REF if name == I.REF else [I.LGBM_DIR])
        runs = {k: T[k].assign(pred=(T[k].pred + G[k].reindex(T[k].index).pred) / 2) for k in RUNS}
        per[name] = np.array([RU.losses(runs[k], caps) for k in RUNS])
        ens[name] = runs[0].assign(pred=np.mean([runs[k].reindex(runs[0].index).pred for k in RUNS], axis=0))
        rows.append(dict(model=R_HYB, inputs=name, runs=len(RUNS), mse_mean=per[name][:, 0].mean(), mse_sd=sd(per[name][:, 0]),
                         mse_har=per[name][:, 0].mean() / mse_har, qlike_mean=per[name][:, 1].mean(), qlike_sd=sd(per[name][:, 1])))
    t = pd.DataFrame(rows)
    ref = pd.read_csv(os.path.join(CSV, "v3_paper_models.csv")).set_index("model").loc[R_HYB]
    assert np.isclose(t.loc[t.inputs == I.REF, "mse_mean"].iloc[0], ref.mse_mean, rtol=1e-9)

    def long(name: str) -> pd.DataFrame:
        e = ens[name][ens[name].index.get_level_values("split") == "test"].reset_index()
        return pd.DataFrame({"dataset": e.dataset, "task": "range", "groupset": "IN", "model": name, "seed": 0,
                             "fold": e.fold, "time": e.time, "y": e.y, "pred": e.pred})

    store = R.Store([long(name) for name in ens])
    for i, r in t.iterrows():
        if r.inputs == I.REF:
            continue
        v = store.pair(("IN", r.inputs), ("IN", I.REF))
        t.loc[i, ["vs_ref", "vs_ref_lo", "vs_ref_hi", "vs_ref_better", "p_ref"]] = [v["ratio"], v["ci_lo"], v["ci_hi"], v["better"], v["p"]]
        t.loc[i, "diff_pct"] = 100 * (r.mse_mean / t.loc[t.inputs == I.REF, "mse_mean"].iloc[0] - 1)
        t.loc[i, "p_runs"] = st.ttest_ind(per[r.inputs][:, 0], per[I.REF][:, 0], equal_var=False).pvalue
    t = _holm(t, ["model"], ["p_ref"])
    t.to_csv(os.path.join(CSV, "v4_inputs.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.round(4).to_string(index=False))


# ============================================================================================ extras
def extras():
    """Descriptive numbers for the text, computed while the manuscript was written (no plan fixed in advance):
    tables/v4_triple_weights.csv, the fitted weights and stacking coefficients of the three-model schemes with the
    real network; tables/v4_confirm.csv, the hybrid with real layers on the 33 days after the sample against its
    network, LightGBM and the quaternion version (averages of three runs, as in paper_v3_hybrid)."""
    import paper_v2_round2 as R
    import paper_v3_hybrid as H
    import paper_v3_triple as TR
    M, _ = P3.build()
    T, G = M[R_NET], M[LGBM]
    rows = []
    for tag, third in (("HAR", M["HAR"][0]), ("HAR-S x", M["HAR-S x"][0])):
        for k in RUNS:
            a = T[k]
            x = pd.DataFrame({"y": a.y, "h": third.reindex(a.index).pred, "g": G[k].reindex(a.index).pred, "n": a.pred})
            for scheme, fn in (("weighted", TR.blend_w), ("stacking", TR.stack)):
                rows += [dict(third=tag, scheme=scheme, run=k, **r) for r in fn(x)[1]]
    w = pd.DataFrame(rows)
    w.to_csv(os.path.join(CSV, "v4_triple_weights.csv"), index=False)
    print(w.groupby(["third", "scheme"])[[c for c in w.columns if c.startswith(("w_", "b_"))]].agg(["min", "max"]).round(3).to_string())
    print((w[(w.scheme == "weighted") & (w.third == "HAR")].w_har == 0).sum(), "of", len(w[(w.scheme == "weighted") & (w.third == "HAR")]), "fits give HAR no weight")

    b = H.base(H.CONFIRM_RUNS)
    te = b[b.split == "test"]
    frame = lambda name, pred: pd.DataFrame({"dataset": te.dataset.to_numpy(), "task": "range", "groupset": "X", "model": name, "seed": 0,
                                             "fold": te.fold.to_numpy(), "time": te.time.to_numpy(), "y": te.y.to_numpy(), "pred": np.asarray(pred)})
    st = R.Store([frame("hybrid, real", (te.R + te.G) / 2), frame("H-LSTM", te.R), frame("LightGBM", te.G), frame("hybrid, quaternion", (te.N + te.G) / 2),
                  frame("H-QLSTM", te.N)])
    rows = []
    for a, o in (("hybrid, real", "H-LSTM"), ("hybrid, real", "LightGBM"), ("hybrid, real", "hybrid, quaternion"), ("hybrid, quaternion", "H-QLSTM"),
                 ("hybrid, quaternion", "LightGBM")):
        v = st.pair(("X", a), ("X", o))
        rows.append(dict(model=a, other=o, ratio=v["ratio"], lo=v["ci_lo"], hi=v["ci_hi"], coins_better=v["better"], p=v["p"]))
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(CSV, "v4_confirm.csv"), index=False)
    print(c.round(4).to_string(index=False))


# ============================================================================================ tables
TEX = {**P3.TEX, R_NET: "H-LSTM", Q_NET: r"\HQ{}", R_HYB: r"\textbf{H-LSTM + LightGBM}", Q_HYB: r"\HQ{} + LightGBM"}
GROUPS_MAIN = [("Linear models, one per coin", ["HAR", "HAR-S x"]), ("Gradient boosting, pooled", [LGBM]),
               ("Networks, pooled", ["DLinear", "CNN, 6 convolutions", "QCNN, 6 convolutions", "PatchTST", "iTransformer", "Transformer", Q_NET, R_NET]),
               ("Hybrids: plain average of the network and LightGBM", [Q_HYB, R_HYB])]


def write(name: str, lines: list[str]):
    os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
    with open(os.path.join(PAPER, "tables", name), "w") as f:
        f.write("% generated by experiments/paper_v4_paper.py tables\n" + "\n".join(lines) + "\n")


GROUPS_EXT = [("Linear, GARCH and range models", ["HAR", "HAR-S x", "Linear, pooled", "GARCH(1,1)", "CARR"]),
              ("Gradient boosting, pooled", [LGBM, "HGB"]),
              ("Networks, pooled", ["DLinear", "CNN, 6 convolutions", "QCNN, 6 convolutions", "PatchTST", "iTransformer", "Transformer", Q_NET, R_NET]),
              ("Plain average of the H-LSTM (or its quaternion version) and LightGBM", [Q_HYB, R_HYB]),
              ("Other averages", ["Five runs of H-LSTM", "H-LSTM + HGB", "Transformer + LightGBM", "H-LSTM + Transformer", "Five runs of H-LSTM + LightGBM",
                                  "H-LSTM + LightGBM + Transformer"])]
TEX_EXT = {"GARCH(1,1)": "GARCH(1,1)", "CARR": "CARR", "Linear, pooled": "Linear on the regressors of LightGBM, pooled", "HGB": "HGB (second library)",
           "Five runs of H-LSTM": "Five runs of H-LSTM", "H-LSTM + HGB": "H-LSTM + HGB", "Transformer + LightGBM": "Transformer + LightGBM",
           "H-LSTM + Transformer": "H-LSTM + Transformer", "Five runs of H-LSTM + LightGBM": "Five runs of H-LSTM + LightGBM",
           "H-LSTM + LightGBM + Transformer": "H-LSTM + LightGBM + Transformer"}
PARAMS_EXT = {"GARCH(1,1)": "5", "CARR": "3", "Linear, pooled": "36", "HGB": "trees"}


def tab_main():
    t = pd.read_csv(os.path.join(CSV, "v4_r2_main.csv")).set_index("model")
    big = {"DLinear", "CNN, 6 convolutions", "QCNN, 6 convolutions"}
    L = [r"\begin{tabular}{@{}lrcccrccrrccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{MSE} & QLIKE & \multicolumn{4}{c}{Hybrid against the model, MSE} & \multicolumn{3}{c}{Hybrid against the model, QLIKE} & MCS \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(lr){6-9}\cmidrule(lr){10-12}\cmidrule(l){13-13}",
         r"Model & Param. & mean\,$\pm$\,sd & $\times$\,HAR & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & coins & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & $p$ \\",
         r"\midrule"]
    for g, (head, names) in enumerate(GROUPS_EXT):
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
        for n in names:
            r = t.loc[n]
            if n in big:
                ql = (f"{r.qlike_mean:.1f}" if r.qlike_mean < 100 else rf"$>$10$^{{3}}$") + r"$^{\dagger}$"
            else:
                ql = pm(r.qlike_mean, r.qlike_sd, 3)
            label = TEX_EXT.get(n) or TEX[n]
            label = {Q_HYB: label + " (quaternion layers)", R_HYB: r"\textbf{H-LSTM + LightGBM (real layers)}"}.get(n, label)
            cells = [label, PARAMS_EXT.get(n, P3.PARAMS.get(n, "")), pm(r.mse_mean, r.mse_sd), f"{r.mse_har:.3f}", ql]
            if n == R_HYB:
                cells += [""] * 7
            else:
                cells += [dp(r.mse_diff_pct, 1, r.mse_verdict == "better"), pv(r.mse_p_runs), pv(r.mse_p_time), f"{int(r.mse_coins_better)}/15",
                          (r"$\dagger$" if n in big else dp(r.qlike_diff_pct, 1, r.qlike_verdict == "better")), pv(r.qlike_p_runs), pv(r.qlike_p_time)]
            cells.append(pv(r.mcs_p) if r.mcs_p < 0.9995 else "1.000")
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("main.tex", L)


def tab_combos():
    c = pd.read_csv(os.path.join(CSV, "v4_combos.csv"))
    c = c[c.version == R_HYB]
    L = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{MSE} & \multicolumn{3}{c}{QLIKE} \\", r"\cmidrule(lr){2-5}\cmidrule(l){6-8}",
         r"Scheme & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{time}}$ \\", r"\midrule"]
    for g, (group, rows) in enumerate(c.groupby("group", sort=False)):
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{group}}}}} \\")
        for r in rows.itertuples():
            first = r.name == R_HYB
            L.append(" & ".join([r.scheme, pm(r.mse_mean, r.mse_sd), "" if first else dp(r.avg_mse_diff_pct), "" if first else pv(r.avg_mse_p_runs),
                                 "" if first else pv(r.avg_mse_p_time), pm(r.qlike_mean, r.qlike_sd, 3), "" if first else dp(r.avg_qlike_diff_pct, 1),
                                 "" if first else pv(r.avg_qlike_p_time)]) + r" \\")
    av = pd.read_csv(os.path.join(CSV, "v4_r2_averages.csv")).set_index("name")
    extra = [("H-LSTM, LightGBM and the Transformer", [("Plain average", "H-LSTM + LightGBM + Transformer")]),
             ("Members chosen on the validation block of each fold, from the eleven single models",
              [("Best pair", "Best pair on the validation block"), ("Best three models", "Best 3 on the validation block"),
               ("Best four models", "Best 4 on the validation block")])]
    for group, rows in extra:
        L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{group}}}}} \\")
        for label, name in rows:
            r = av.loc[name]
            L.append(" & ".join([label, pm(r.mse_mean, r.mse_sd), dp(r.hyb_mse_diff_pct), pv(r.hyb_mse_p_runs), pv(r.hyb_mse_p_time),
                                 pm(r.qlike_mean, r.qlike_sd, 3), dp(r.hyb_qlike_diff_pct, 1), pv(r.hyb_qlike_p_time)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("combos.tex", L)


def tab_partners():
    p = pd.read_csv(os.path.join(CSV, "v4_partners.csv"))
    lab = {Q_NET: r"\HQ{}", R_NET: "H-LSTM", LGBM: "LightGBM", "HAR-S x": r"HAR-S$\times$"}
    name = lambda a, b: (f"Two runs of {lab.get(a, a)}" if a == b else f"{lab.get(a, a)} + {lab.get(b, b)}")
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & Error & \multicolumn{4}{c}{MSE} & QLIKE \\", r"\cmidrule(lr){3-6}\cmidrule(l){7-7}",
         r"Members of the average & corr. & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & mean \\", r"\midrule"]
    for g, (head, rows) in enumerate([("With LightGBM", p[(p.b == LGBM)]), ("Without LightGBM", p[(p.b != LGBM)])]):
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
        for r in rows.itertuples():
            first = (r.a, r.b) == (R_NET, LGBM)
            L.append(" & ".join([name(r.a, r.b) + (" (the hybrid)" if first else ""), f"{r.corr:.3f}", pm(r.mse_mean, r.mse_sd),
                                 "" if first else dp(r.hyb_mse_diff_pct), "" if first else pv(r.hyb_mse_p_runs), "" if first else pv(r.hyb_mse_p_time),
                                 f"{r.qlike_mean:.3f}"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("partners.tex", L)


def tab_inputs():
    from scipy import stats as st
    x = pd.read_csv(os.path.join(CSV, "v3_inputs.csv"))
    h = pd.read_csv(os.path.join(CSV, "v4_inputs.csv"))
    order = ["LightGBM", "H-QLSTM", "H-LSTM (real twin)", R_HYB]
    head = {"LightGBM": "LightGBM", "H-LSTM (real twin)": "H-LSTM", "H-QLSTM": r"\HQ{}", R_HYB: "H-LSTM + LightGBM"}
    lab = {"price, market, calendar": "Price, market, calendar (reference)", "without market": r"\quad without market", "without calendar": r"\quad without calendar",
           "+ technical": r"\quad + technical", "+ social": r"\quad + social", "+ sentiment": r"\quad + sentiment",
           "+ social, sentiment": r"\quad + social and sentiment", "+ technical, social, sentiment": r"\quad + technical, social, sentiment"}
    L = [r"\begin{tabular}{@{}l" + "cr" * 4 + r"@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{head[m]}}}" for m in order) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4)),
         "Inputs & " + " & ".join([r"MSE & $\Delta$ (\%)"] * 4) + r" \\", r"\midrule"]
    ref = "price, market, calendar"
    for inp in lab:
        cells = [lab[inp]]
        for m in order:
            src = h if m == R_HYB else x
            r = src[(src.model == m) & (src.inputs == inp)].iloc[0]
            if inp == ref:
                cells += [pm(r.mse_mean, r.mse_sd), ""]
                continue
            base = src[(src.model == m) & (src.inputs == ref)].iloc[0]
            # over runs: Welch's t-test from the means and standard deviations of the five runs of each input set;
            # over time: the DM test of paper_v3_inputs (forecasts averaged over the runs), Holm over the seven sets
            p_runs = st.ttest_ind_from_stats(r.mse_mean, r.mse_sd, int(r.runs), base.mse_mean, base.mse_sd, int(base.runs), equal_var=False).pvalue
            star = "$^{*}$" if (p_runs < 0.05 and r.p_ref_holm < 0.05) else ""
            cells += [pm(r.mse_mean, r.mse_sd), dp(100 * (r.mse_mean / base.mse_mean - 1), 1) + star]
        L.append(" & ".join(cells) + r" \\")
        if inp == ref:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("inputs.tex", L)


def tab_robust():
    b = pd.read_csv(os.path.join(CSV, "v4_bars.csv"))
    b = b[b.ours == R_HYB]
    lab = {"HAR": "HAR", "HAR-S x": r"HAR-S$\times$", LGBM: "LightGBM", R_NET: "H-LSTM", Q_NET: r"\HQ{}",
           R_HYB: "H-LSTM + LightGBM", Q_HYB: r"\HQ{} + LightGBM"}
    names = {"1h": "Hourly bars", "4h": "Four-hour bars", "1d": "Daily bars"}
    order = ["HAR", "HAR-S x", LGBM, Q_NET, R_NET, Q_HYB, R_HYB]
    L = [r"\begin{tabular}{@{}lcccrcccrcc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{MSE} & QLIKE & \multicolumn{4}{c}{Hybrid against the model, MSE} & \multicolumn{3}{c}{Hybrid against the model, QLIKE} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-8}\cmidrule(l){9-11}",
         r"Model & mean\,$\pm$\,sd & $\times$\,HAR & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & coins & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ \\",
         r"\midrule"]
    for g, freq in enumerate(names):
        rows = b[b.bars == freq].set_index("model")
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{11}}{{@{{}}l}}{{\emph{{{names[freq]}}} ({int(rows.test_coin_bars.iloc[0]):,} test forecasts, {int(rows.runs.max())} runs)}} \\")
        for model in order:
            r = rows.loc[model]
            cells = [lab[model], pm(r.mse_mean, r.mse_sd), f"{r.mse_har:.3f}", pm(r.qlike_mean, r.qlike_sd, 3)]
            if model == R_HYB:
                cells += [""] * 7
            else:
                cells += [dp(r.mse_diff_pct, 1, r.mse_verdict == "better"), pv(r.mse_p_runs), pv(r.mse_p_time), f"{int(r.mse_coins_better)}/15",
                          dp(r.qlike_diff_pct, 1, r.qlike_verdict == "better"), pv(r.qlike_p_runs), pv(r.qlike_p_time)]
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("robust.tex", L)


def tab_limits():
    t = pd.read_csv(os.path.join(CSV, "v4_limits.csv"))
    t = t[t.ours == R_HYB].set_index("model")
    cov = pd.read_csv(os.path.join(CSV, "v4_r2_coverage.csv")).set_index("model")
    L = [r"\begin{tabular}{@{}lcccccccccc@{}}", r"\toprule",
         r" & Exceedance & \multicolumn{2}{c}{Coins rejecting} & Limit & \multicolumn{2}{c}{Tick loss} & \multicolumn{4}{c}{Hybrid against the model, tick loss} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}\cmidrule(l){8-11}",
         r"Model & (\%) & UC & CC & $\times$\,HAR & mean\,$\pm$\,sd ($\times 10^{2}$) & $\times$\,HAR & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & coins \\",
         r"\midrule"]
    for g, (head, names) in enumerate(GROUPS_MAIN):
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{11}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
        for n in names:
            r = t.loc[n]
            label = {Q_HYB: TEX[n] + " (quaternion layers)", R_HYB: r"\textbf{H-LSTM + LightGBM (real layers)}"}.get(n, TEX[n])
            cells = [label, f"{100 * r.exceed_mean:.2f}", f"{cov.loc[n, 'coins_reject_uc']:.1f}", f"{cov.loc[n, 'coins_reject_cc']:.1f}",
                     f"{r.limit_har_mean:.3f}", pm(100 * r.tick_mean, 100 * r.tick_sd, 3), f"{r.tick_har:.3f}"]
            cells += [""] * 4 if n == R_HYB else [dp(r.diff_pct, 1, r.verdict == "better"), pv(r.p_runs), pv(r.p_time), f"{int(r.coins_better)}/15"]
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("limits.tex", L)


def tab_new(tag: str = "newcoins"):
    """The frozen system outside the design (tables/v4_<tag>.csv and v4_<tag>_limits.csv): the 15 coins outside the
    study on the test blocks of the paper (newcoins), and both sets of coins on the earlier period (early_study, early_new)."""
    t = pd.read_csv(os.path.join(CSV, f"v4_{tag}.csv"))
    t = t[t.ours == R_HYB].set_index("model")
    lim = pd.read_csv(os.path.join(CSV, f"v4_{tag}_limits.csv"))
    lim = lim[lim.ours == R_HYB].set_index("model")
    lab = {"HAR": "HAR", "HAR-S x": r"HAR-S$\times$", "LightGBM": "LightGBM", "Transformer": "Transformer", "H-QLSTM": r"\HQ{}", "H-LSTM": "H-LSTM",
           Q_HYB: r"\HQ{} + LightGBM (quaternion layers)", R_HYB: r"\textbf{H-LSTM + LightGBM (real layers)}"}
    groups = [("Single models", ["HAR", "HAR-S x", "LightGBM", "Transformer", "H-QLSTM", "H-LSTM"]), ("Hybrids", [Q_HYB, R_HYB]),
              ("Other averages", ["Transformer + LightGBM", "H-LSTM + Transformer", "H-LSTM + LightGBM + Transformer"])]
    L = [r"\begin{tabular}{@{}lcccrcccrcc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{MSE} & QLIKE & \multicolumn{4}{c}{Hybrid against the model, MSE} & \multicolumn{3}{c}{Hybrid against the model, tick loss} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-8}\cmidrule(l){9-11}",
         r"Model & mean\,$\pm$\,sd & $\times$\,HAR & mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & coins & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ \\",
         r"\midrule"]
    for g, (head, names) in enumerate(groups):
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{11}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
        for n in names:
            r = t.loc[n]
            cells = [lab.get(n, n), pm(r.mse_mean, r.mse_sd), f"{r.mse_har:.3f}", pm(r.qlike_mean, r.qlike_sd, 3)]
            if n == R_HYB:
                cells += [""] * 7
            else:
                cells += [dp(r.mse_diff_pct, 1, r.mse_verdict == "better"), pv(r.mse_p_runs), pv(r.mse_p_time), f"{int(r.mse_coins_better)}/15"]
                if n in lim.index:
                    x = lim.loc[n]
                    cells += [dp(x.diff_pct, 1, x.verdict == "better"), pv(x.p_runs), pv(x.p_time)]
                else:
                    cells += [""] * 3
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(f"{tag}.tex", L)


def tab_confirm():
    """The frozen system on the three held-out data sets, side by side (tables/v4_newcoins.csv, v4_early_study.csv,
    v4_early_new.csv; the averages of several runs of the network from tables/v4_r4_averages.csv)."""
    sets = [("newcoins", "New coins, test period of the study"), ("early_study", "Coins of the study, earlier period"), ("early_new", "New coins, earlier period")]
    T = {tag: (lambda t: t[t.ours == R_HYB].set_index("model"))(pd.read_csv(os.path.join(CSV, f"v4_{tag}.csv"))) for tag, _ in sets}
    lab = {"HAR": "HAR", "HAR-S x": r"HAR-S$\times$", "LightGBM": "LightGBM", "Transformer": "Transformer", "H-QLSTM": r"\HQ{}", "H-LSTM": "H-LSTM",
           Q_HYB: r"\HQ{} + LightGBM", R_HYB: r"\textbf{H-LSTM + LightGBM}"}
    groups = [("Single models", ["HAR", "HAR-S x", "LightGBM", "Transformer", "H-QLSTM", "H-LSTM"]), ("Hybrids", [Q_HYB, R_HYB]),
              ("Other averages", ["Two runs of H-LSTM", "Five runs of H-LSTM", "Transformer + LightGBM", "H-LSTM + Transformer",
                                  "Five runs of H-LSTM + LightGBM", "H-LSTM + LightGBM + Transformer"])]
    av = pd.read_csv(os.path.join(CSV, "v4_r4_averages.csv"))
    av = av[av.a == R_HYB].set_index(["data", "b"])
    L = [r"\begin{tabular}{@{}l" + "crcc" * 3 + r"@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{name}}}" for _, name in sets) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(3)),
         "Model & " + " & ".join([r"MSE, mean\,$\pm$\,sd & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$"] * 3) + r" \\", r"\midrule"]
    for g, (head, names) in enumerate(groups):
        if g:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
        for n in names:
            cells = [lab.get(n, n)]
            for tag, _ in sets:
                if n not in T[tag].index:                  # averages of the fourth round
                    r = av.loc[(tag, n)]
                    cells += [pm(r.b_mean, r.b_sd), dp(r.diff_pct, 1, r.verdict == "better"), pv(r.p_runs), pv(r.p_time)]
                    continue
                r = T[tag].loc[n]
                assert (tag, n) not in av.index or np.isclose(av.loc[(tag, n)].b_mean, r.mse_mean, rtol=1e-9), n
                cells.append(pm(r.mse_mean, r.mse_sd))
                cells += ["", "", ""] if n == R_HYB else [dp(r.mse_diff_pct, 1, r.mse_verdict == "better"), pv(r.mse_p_runs), pv(r.mse_p_time)]
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("confirm.tex", L)


def tab_tuned():
    """Every family at the setting of the paper and at the setting chosen on the validation block (tables/v4_tuned.csv)."""
    t = pd.read_csv(os.path.join(CSV, "v4_tuned.csv")).set_index("family")
    n_set = pd.read_csv(os.path.join(CSV, "v4_tuned_settings.csv")).groupby("family").size()
    lab = {"H-QLSTM": r"\HQ{}"}
    L = [r"\begin{tabular}{@{}lccccrccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{MSE} & \multicolumn{4}{c}{Tuned hybrid against the tuned model} \\", r"\cmidrule(lr){3-5}\cmidrule(l){6-9}",
         r"Model & Settings & default & tuned, mean\,$\pm$\,sd & change (\%) & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ & coins \\", r"\midrule"]
    for n in ("HAR", "HAR-S x", "LightGBM", "DLinear", "CNN", "QCNN", "PatchTST", "iTransformer", "Transformer", "H-QLSTM", "H-LSTM"):
        r = t.loc[n]
        fixed = n in ("HAR", "HAR-S x")
        gain = "" if fixed else dp(r.gain_mse_diff_pct, 1) + ("$^{*}$" if r.gain_mse_verdict in ("better", "worse") else "")
        L.append(" & ".join([{"HAR-S x": r"HAR-S$\times$"}.get(n, lab.get(n, n)), "1" if fixed else str(int(n_set[n])), f"{r.default_mse:.4f}",
                             f"{r.tuned_mse:.4f}" if fixed else pm(r.tuned_mse, r.tuned_sd), gain, dp(r.tunedhyb_mse_diff_pct, 1, r.tunedhyb_mse_verdict == "better"),
                             pv(r.tunedhyb_mse_p_runs), pv(r.tunedhyb_mse_p_time), f"{int(r.tunedhyb_mse_coins_better)}/15"]) + r" \\")
    L.append(r"\addlinespace[2pt]")
    h, d = t.loc["Hybrid (tuned)"], t.loc["Hybrid (default)"]
    L.append(" & ".join([r"\textbf{H-LSTM + LightGBM}", "", f"{d.tuned_mse:.4f}", pm(h.tuned_mse, h.tuned_sd), dp(100 * (h.tuned_mse / d.tuned_mse - 1), 1), "", "", "", ""]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tuned.tex", L)


def tab_views():
    """The two-by-two of learner and input form (tables/v4_r3_views.csv)."""
    t = pd.read_csv(os.path.join(CSV, "v4_r3_views.csv"))
    short = {"network, window (H-LSTM)": "LSTM on the window (H-LSTM)", "tree, summaries (LightGBM)": "Trees on the summaries (LightGBM)", "tree, window": "Trees on the window",
             "network, summaries": "Network on the summaries"}
    L = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r" & & & Error & Gain & \multicolumn{3}{c}{Against H-LSTM + LightGBM} \\", r"\cmidrule(l){6-8}",
         r"Models & What differs & MSE, mean\,$\pm$\,sd & corr. & (\%) & $\Delta$ (\%) & $p_{\mathrm{runs}}$ & $p_{\mathrm{time}}$ \\", r"\midrule"]
    for r in t[t.kind == "single"].itertuples():
        L.append(" & ".join([short[r.name], "", pm(r.mse_mean, r.mse_sd), "", "", "", "", ""]) + r" \\")
    L.append(r"\addlinespace[2pt]")
    order = {"both": 2, "view only": 0, "learner only": 1}
    label = {"both": "model class and input", "view only": "input only", "learner only": "model class only"}
    for r in sorted(t[t.kind != "single"].itertuples(), key=lambda r: (order[r.kind], r.mse_mean)):
        a, b = r.name.split(" + ")
        ours = abs(r.hyb_mse_diff_pct) < 1e-9
        L.append(" & ".join([short[a].split(" (")[0] + " + " + short[b].split(" (")[0].lower(), label[r.kind], pm(r.mse_mean, r.mse_sd), f"{r.corr:.3f}", f"{r.gain_pct:.2f}",
                             "" if ours else dp(r.hyb_mse_diff_pct), "" if ours else pv(r.hyb_mse_p_runs), "" if ours else pv(r.hyb_mse_p_time)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("views.tex", L)


def tab_levels():
    """Range limits at three levels (tables/v4_r3_limits.csv)."""
    t = pd.read_csv(os.path.join(CSV, "v4_r3_limits.csv"))
    lab = {"HAR": "HAR", "HAR-S x": r"HAR-S$\times$", LGBM: "LightGBM", "LightGBM, quantile objective": "LightGBM, quantile objective", R_NET: "H-LSTM",
           "Transformer": "Transformer", R_HYB: r"\textbf{H-LSTM + LightGBM}", "H-LSTM + LightGBM + Transformer": "H-LSTM + LightGBM + Transformer"}
    levels = (0.95, 0.99, 0.995)
    L = [r"\begin{tabular}{@{}l" + "ccr" * 3 + r"@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{100 * q:g}\%}}" for q in levels) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(3)),
         "Model & " + " & ".join([r"exc. (\%) & tick $\times$\,HAR & $\Delta$ (\%)"] * 3) + r" \\", r"\midrule"]
    for n in lab:
        cells = [lab[n]]
        for q in levels:
            r = t[(t.level == q) & (t.model == n)]
            if not len(r):
                cells += ["", "", ""]
                continue
            r = r.iloc[0]
            cells += [f"{100 * r.exceed:.2f}", f"{r.tick_har:.3f}", "" if n == R_HYB else dp(r.diff_pct, 1, r.verdict == "better")]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("levels.tex", L)


def tables():
    for fn in (tab_main, tab_combos, tab_partners, tab_inputs, tab_robust, tab_limits, tab_new, tab_tuned, tab_views, tab_levels):
        fn()
    tab_confirm()
    shutil.copy(os.path.join(PAPER_V3, "tables", "datasets.tex"), os.path.join(PAPER, "tables", "datasets.tex"))
    for f in sorted(os.listdir(os.path.join(PAPER, "tables"))):
        print(f)


# ============================================================================================ figures
SHORT = {**P3.SHORT, R_NET: "H-LSTM", Q_NET: "H-QLSTM", R_HYB: "H-LSTM + LightGBM", Q_HYB: "H-QLSTM + LightGBM"}
FAMILY = {"HAR": (GREY, "s"), "HAR-S x": (GREY, "s"), LGBM: (ORANGE, "D"), R_HYB: (RED, "o"), Q_HYB: (RED, "o")}      # networks: blue circles


def _style():
    plt = P3._style()
    os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
    return plt


def fig_main():
    """(a) MSE of the five runs of every model; (b) the hybrid against the closest models; (c) MSE per coin."""
    plt = _style()
    import paper_v2_round2 as R
    from paper_v3_tests import bar_losses
    m = pd.read_csv(os.path.join(CSV, "v3_paper_models.csv")).set_index("model")
    t = pd.read_csv(os.path.join(CSV, "v4_tests.csv"))
    t = t[t.ours == R_HYB].set_index("other")
    coins = pd.read_csv(os.path.join(CSV, "v3_paper_coins.csv"), index_col=0)
    M, _ = P3.build()
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
    close = [LGBM, "PatchTST", "iTransformer", "Transformer", Q_NET, R_NET, Q_HYB]
    for y, n in enumerate(order):
        if n not in close:
            if n != R_HYB:
                ax.annotate(f"{t.loc[n, 'mse_diff_pct']:.1f}%", (-6.35, y), ha="left", va="center", fontsize=6.3, color="0.35")
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
    rel = coins.div(coins["HAR"], axis=0).sort_values(R_HYB, ascending=False)
    for col, colour, marker, label in ((LGBM, ORANGE, "D", "LightGBM"), (R_NET, BLUE, "o", "H-LSTM"), (R_HYB, RED, "o", "hybrid")):
        ax.plot(rel[col], range(len(rel)), ls="", marker=marker, ms=3.6, color=colour, mec="white", mew=0.4, label=label)
    for y, (lo, hi) in enumerate(zip(rel[[LGBM, R_NET, R_HYB]].min(axis=1), rel[[LGBM, R_NET, R_HYB]].max(axis=1))):
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
    names = ["HAR", "HAR-S x", LGBM, "PatchTST", "iTransformer", "Transformer", Q_NET, R_NET]
    corr = corr.loc[names, names]
    w = pd.read_csv(os.path.join(CSV, "v3_paper_weight.csv"))
    fitted = pd.read_csv(os.path.join(CSV, "v3_paper_weights.csv"))
    tm = pd.read_csv(os.path.join(CSV, "v4_time.csv"), index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw=dict(width_ratios=[1.05, 0.9, 1.15], wspace=0.42))
    ax = axes[0]
    v = corr.to_numpy()
    ax.imshow(np.where(np.isnan(v), 1.0, v), cmap="Blues", vmin=0.88, vmax=1.0)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(v[i, j]):
                ax.text(j, i, f"{v[i, j]:.3f}"[1:], ha="center", va="center", fontsize=5.0, color="white" if v[i, j] > 0.962 else "0.15")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=5.6, color="white")
    lab = [SHORT[n] for n in names]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(lab, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(lab)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("(a) Correlation of forecast errors", loc="left")
    ax = axes[1]
    for net, colour in ((R_NET, BLUE), (Q_NET, "0.6")):
        for split, ls in (("test", "-"), ("val", "--")):
            g = w[(w.network == net) & (w.split == split)]
            ax.plot(g.w, g.mse, color=colour, ls=ls, lw=1.1 if net == R_NET else 0.7,
                    label={(R_NET, "test"): "test blocks", (R_NET, "val"): "validation blocks", (Q_NET, "test"): "quaternion layers"}.get((net, split)))
    f = fitted[fitted.scheme == "W2 twin"].weight_on_network
    ax.axvspan(f.min(), f.max(), color=ORANGE, alpha=0.22, lw=0)
    ax.axvline(0.5, color=RED, lw=0.8)
    ymax = w[(w.network == R_NET)].mse.max()
    ax.text(0.49, ymax, "plain\naverage", ha="right", va="top", fontsize=6.3, color=RED)
    ax.text(f.max() + 0.02, ymax - 0.0004, "fitted\nweights", ha="left", va="top", fontsize=6.3, color="#9a6a00")
    ax.set_xlabel("weight of the network")
    ax.set_ylabel("MSE")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0\nLightGBM", "0.25", "0.5", "0.75", "1\nnetwork"])
    ax.set_title("(b) MSE against the weight", loc="left")
    ax = axes[2]
    for col, colour, ls, label in (("hybrid_lgbm_mse", ORANGE, "-", "LightGBM"), ("hybrid_net_mse", BLUE, "-", "H-LSTM"),
                                   ("hybrid_transformer_mse", GREEN, "-", "Transformer"), ("hybrid_quaternion_mse", RED, "--", "quaternion version")):
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


def fig_limits(zoom_days: int = 21):
    """(a) the range of BTC with the 95% limits of HAR and of the hybrid; (b) tick loss of every model relative to HAR."""
    plt = _style()
    import matplotlib.dates as mdates
    ex = pd.read_csv(os.path.join(CSV, "v4_limits_example.csv"), index_col=0, parse_dates=True)
    t = pd.read_csv(os.path.join(CSV, "v4_limits.csv"))
    t = t[t.ours == R_HYB].set_index("model")
    peak = ex.y.idxmax()
    z = ex[(ex.index >= peak - pd.Timedelta(days=zoom_days * 2 / 3)) & (ex.index <= peak + pd.Timedelta(days=zoom_days / 3))]
    pct = lambda s: 100 * np.exp(s)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.42))
    ax = axes[0]
    ax.bar(z.index, pct(z.y), width=0.13, color="0.72", lw=0, label="realised range")
    ax.step(z.index, pct(z.limit_har), where="mid", color=GREY, lw=0.9, ls="--", label="limit from HAR")
    ax.step(z.index, pct(z.limit_hybrid), where="mid", color=RED, lw=1.0, label="limit from the hybrid")
    over = z[z.y > z.limit_hybrid]
    ax.plot(over.index, pct(over.y), ls="", marker="v", ms=3.2, color=RED, mec="white", mew=0.3, label="range above the hybrid's limit")
    ax.set_ylabel("range of the four-hour bar (%)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.1)
    ax.set_title(f"(a) BTC, {z.index[0]:%d %b} to {z.index[-1]:%d %b %Y}", loc="left")
    ax = axes[1]
    order = MAIN[::-1]
    for y, n in enumerate(order):
        colour, marker = FAMILY.get(n, (BLUE, "o"))
        ax.plot(t.loc[n, "tick_har"], y, ls="", marker=marker, ms=4.2, color=colour, mec="white", mew=0.5)
        ax.text(1.012, y, f"{100 * t.loc[n, 'exceed_mean']:.1f}%", va="center", ha="left", fontsize=6.3, color="0.25")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[n] for n in order])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(0.905, 1.045)
    ax.set_xlabel("tick loss of the 95% limit relative to HAR")
    ax.set_title("(b) Tick loss; exceedance rate at right", loc="left")
    ax.grid(axis="x", color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(PAPER, "figures", "fig_limits.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_controls():
    """Design controls of the network (three-seed averages; ten where noted), as in manuscript v3 with the labels of v4:
    each control against the reference network (quaternion layers), and the quaternion against the real version in every control."""
    plt = _style()
    g = pd.read_csv(os.path.join(CSV, "v3_grid.csv"))
    sm = pd.read_csv(os.path.join(CSV, "v3_small.csv")).set_index("pair")
    se = pd.read_csv(os.path.join(CSV, "v3_grid_seeds.csv")).set_index("pair")
    bars = pd.read_csv(os.path.join(CSV, "v3_grid_bars.csv"))
    cr = pd.read_csv(os.path.join(CSV, "v4_r2_controls.csv")).set_index("control")       # the controls with real layers against the real reference
    rows = [("Reference network, 20,000 parameters (10 seeds)", None, (se.loc["Candidate A / real twin", "ratio_all"], se.loc["Candidate A / real twin", "ci_lo"], se.loc["Candidate A / real twin", "ci_hi"]))]
    label = {"without the range in the price quaternion": "Inputs: price without the range", "without the calendar group": "Inputs: without calendar",
             "without the market group": "Inputs: without market", "with social and sentiment groups": "Inputs: + social and sentiment",
             "additive term instead of a group": "Calendar as an additive term", "concatenation": "Fusion: concatenation", "dense": "Fusion: dense layer",
             "two layers": "Two recurrent layers", "flat QLSTM / LSTM": "One encoder for all groups", "5,000 parameters": "5,000 parameters",
             "80,000 parameters": "80,000 parameters", "3e-3": "Learning rate 3e-3", "3e-4": "Learning rate 3e-4", "one model per coin": "One model per coin"}
    for r in g.itertuples():
        if r.control in label:
            c = cr.loc[r.control]
            rows.append((label[r.control], (c.real_vs_ref, c.real_lo, c.real_hi), (r.twin_ratio, r.twin_ci_lo, r.twin_ci_hi)))
        if r.control == "flat QLSTM / LSTM":
            for n in ("500", "1,000", "2,000"):
                a, b = sm.loc[f"real twin, {n} / 20,000 parameters"], sm.loc[f"{n} parameters: H-QLSTM / real twin"]
                rows.append((f"{n} parameters (10 seeds)", (a.ratio_all, a.ci_lo, a.ci_hi), (b.ratio_all, b.ci_lo, b.ci_hi)))
    for freq, name in (("1h", "Hourly bars"), ("1d", "Daily bars")):
        b = bars[(bars.bars == freq) & (bars.model == "Candidate A, real twin")].iloc[0]
        rows.append((name, None, (b.A_over_model, b.ci_lo, b.ci_hi)))
    rows = rows[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.06))
    for ax, k, colour, title, xlim in ((axes[0], 1, BLUE, "(a) Control against the reference network, real layers", (0.985, 1.05)),
                                       (axes[1], 2, RED, "(b) Quaternion against real layers", (0.988, 1.027))):
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
    _style()
    shutil.copy(os.path.join(PAPER_V3, "figures", "fig_data.pdf"), os.path.join(PAPER, "figures", "fig_data.pdf"))       # unchanged from manuscript v3
    for fn in (fig_main, fig_combination, fig_limits, fig_controls):
        fn()
        print(fn.__name__)


if __name__ == "__main__":
    fns = {"stats": stats, "limits": limits, "bars": bars, "inputs": inputs, "extras": extras, "tables": tables, "figs": figs}
    for a in sys.argv[1:] or ["stats"]:
        fns[a]()
