"""
Fourth round for manuscript v4: checks asked for by the re-review and the fresh review of 2026-11-24 (after the third
round). Everything is computed from saved forecasts; no model is fitted.

Plan (2026-11-24, written and committed before any number below is computed). Known before: every table of the
manuscript and of rounds two and three. "The hybrid" is H-LSTM + LightGBM (run k with run k). Tests and rule as in
the paper: Welch over five runs (one-sample against a single forecast), Diebold-Mariano over time on the loss
difference averaged over runs and coins; better or worse only if both p < 0.05. All analyses are secondary and
descriptive: none changes the primary rule of a confirmation, and each is reported whatever it shows.

1. averages   tables/v4_r4_averages.csv. On the four data sets (design data; new coins in the test period; coins of
              the study and new coins in the earlier period): the hybrid against averages of the same or a higher
              cost: two runs of the H-LSTM (run k with run k+1), the five-run average of the H-LSTM, two runs of the
              H-LSTM with LightGBM (thirds), five runs of the H-LSTM with LightGBM (halves), and the three-member
              average with the Transformer; and the three-member average against the two averages with LightGBM.
              Reading: the text says "the hybrid is better than a homogeneous average of the same size" for a data
              set only where the hybrid is better than two runs of the H-LSTM by the rule; "a third model of another
              kind adds more than further runs of the network" only where the three-member average is better than the
              averages with two or five runs of the network by the rule.
2. overlap    tables/v4_r4_overlap.csv. In the design runs the last 180 days of the earlier period were the validation
              block of the first fold of the coins of the study (early stopping, fitted weights). The earlier period of
              the coins of the study without these days: the hybrid against LightGBM and against the H-LSTM.
3. bandwidth  tables/v4_r4_bandwidth.csv. The two primary comparisons on the four data sets with Bartlett lags of 14
              (the paper), 42 and 84 bars, with 90% and 95% intervals.
4. print      tables/v4_r4_print.csv. Design data without the bar of 10 October 2025 20:00 UTC, the bar before it and
              the 43 bars after it (every forecast whose target or window contains the bar): the hybrid against HAR,
              LightGBM and the H-LSTM. And the number of test targets at the floor of the range.
5. flat       tables/v4_r4_flat.csv. The hybrid whose network is one LSTM for all 12 features (the control of the
              design study, runs 0-2) against the hybrid of the paper on the same runs, and the two networks alone.
              Reading: the statement that the gain does not need one encoder per group stays for the hybrid only if
              the hybrid with the single LSTM is not worse by the rule.
6. gains      tables/v4_r4_gains.csv. The gains of the two-by-two (tables/v4_r3_views.csv) per run: mean and standard
              deviation over the five runs for each pair, and the difference between the mean of the two pairs that
              differ in the kind of model only and the mean of the two that differ in the input only, with a t interval
              over the runs. Reading: "about equally" stays if the 95% interval of the difference contains zero.

    .venv/bin/python -W ignore experiments/paper_v4_round4.py [averages|overlap|bandwidth|print|flat|gains|all]
"""

import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

CSV, SEEDS = "docs/paper/manuscript_v2/tables", (0, 1, 2, 3, 4)
SETS = ("design", "newcoins", "early_study", "early_new")
HAR, LGBM, NET, TRF, HYB, TRIPLE = "HAR", "LightGBM", "H-LSTM", "Transformer", "H-LSTM + LightGBM", "H-LSTM + LightGBM + Transformer"
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
mix = lambda *fs: fs[0].assign(pred=sum(f.reindex(fs[0].index).pred for f in fs) / len(fs))


def _set(tag: str) -> tuple[dict, pd.Series]:
    """Forecasts (model -> run -> frame of y and pred) and QLIKE caps of one of the four data sets."""
    import paper_v3_runs as RU
    if tag == "design":
        import paper_v2_round2 as R
        import paper_v3_paper as P3
        M0, _ = P3.build()
        return {HAR: M0["HAR"], LGBM: M0[P3.LGBM], NET: M0[P3.TWIN], TRF: M0["Transformer"]}, R.train_caps()
    import paper_v4_newcoins as N
    importlib.reload(N)
    if tag.startswith("early"):
        import paper_v4_round3 as R3
        N = R3._early(tag.split("_")[1])
    fr = N.frames()
    caps = pd.DataFrame([dict(dataset=ds, fold=k, cap=float(np.exp(2 * np.nanmax(d.frame["y_range"].to_numpy(float)[fo["train"]]))))
                         for ds, (d, folds) in fr.items() for k, fo in enumerate(folds)]).set_index(["dataset", "fold"]).cap
    get = RU.forecasts
    M = {HAR: {0: get(("P", "har"), [N.BENCH])[0]}, LGBM: get(("PM", "lgbmsl-pool"), [N.BENCH]), NET: get(("PRMC", "hl_gattn-pool"), [N.OUT_RUNS]),
         TRF: get(("PRMC", "transformer-pool"), [N.OUT_RUNS])}
    assert all(sorted(M[n]) == list(SEEDS) for n in (LGBM, NET, TRF)), tag
    return M, caps


def _bars(M: dict, caps: pd.Series) -> dict:
    """model -> list over runs of the per-bar losses (mse, qlike), indexed by (time, dataset)."""
    from paper_v3_tests import bar_losses
    return {n: [bar_losses(f, caps) for f in r.values()] for n, r in M.items()}


def _dm(series: np.ndarray, lag: int | None = None) -> tuple[float, float]:
    """Diebold-Mariano statistic and two-sided p-value of a mean of zero, Bartlett kernel (src.paper.evaluate with a free lag)."""
    n = len(series)
    lag = int(np.floor(n ** (1 / 3))) if lag is None else lag
    dc = series - series.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = series.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def _cmp(B: dict, a: str, b: str, loss: str = "mse", keep=None, lag: int | None = None, runs=None) -> dict:
    """The two tests of the paper for model a against model b, on the bars that `keep` selects and the given runs."""
    from paper_v3_tests import verdict
    sel = lambda x: x[loss] if keep is None else x[loss][keep(x.index)]
    ra = B[a] if runs is None or len(B[a]) == 1 else [B[a][k] for k in runs]
    rb = B[b] if runs is None or len(B[b]) == 1 else [B[b][k] for k in runs]
    xa, xb = np.array([sel(x).mean() for x in ra]), np.array([sel(x).mean() for x in rb])
    if len(xa) > 1 and len(xb) > 1:
        p_t = stats.ttest_ind(xa, xb, equal_var=False).pvalue
    else:
        p_t = stats.ttest_1samp(xa if len(xa) > 1 else xb, (xb if len(xa) > 1 else xa)[0]).pvalue
    d = (sum(sel(x) for x in ra) / len(ra) - sum(sel(x) for x in rb) / len(rb)).unstack("dataset")
    series = d.mean(axis=1).sort_index().to_numpy()
    assert np.isclose(series.mean(), xa.mean() - xb.mean(), rtol=1e-6, atol=1e-9), (a, b)
    stat, p_dm = _dm(series, lag)
    se, pct = abs(series.mean() / stat), 100 / xb.mean()
    return dict(a=a, b=b, a_mean=xa.mean(), a_sd=sd(xa), b_mean=xb.mean(), b_sd=sd(xb), diff_pct=100 * (xa.mean() / xb.mean() - 1),
                lo90=pct * (series.mean() - 1.645 * se), hi90=pct * (series.mean() + 1.645 * se), lo95=pct * (series.mean() - 1.96 * se),
                hi95=pct * (series.mean() + 1.96 * se), p_runs=p_t, p_time=p_dm, coins_better=int((d.mean(axis=0) < 0).sum()), bars=len(series),
                verdict=verdict(xa.mean() - xb.mean(), p_t, p_dm))


def _with_averages(M: dict) -> dict:
    N_ = M[NET]
    X = dict(M)
    X[HYB] = {k: mix(N_[k], M[LGBM][k]) for k in SEEDS}
    X[TRIPLE] = {k: mix(N_[k], M[LGBM][k], M[TRF][k]) for k in SEEDS}
    X["Two runs of H-LSTM"] = {k: mix(N_[k], N_[(k + 1) % 5]) for k in SEEDS}
    X["Five runs of H-LSTM"] = {0: mix(*[N_[k] for k in SEEDS])}
    X["Two runs of H-LSTM + LightGBM"] = {k: mix(N_[k], N_[(k + 1) % 5], M[LGBM][k]) for k in SEEDS}
    mean5 = X["Five runs of H-LSTM"][0]
    X["Five runs of H-LSTM + LightGBM"] = {k: mix(mean5, M[LGBM][k]) for k in SEEDS}
    return X


_CACHE: dict = {}


def _loaded(tag: str) -> dict:
    if tag not in _CACHE:
        M, caps = _set(tag)
        _CACHE[tag] = _bars(_with_averages(M), caps)
    return _CACHE[tag]


def averages():
    rows = []
    for tag in SETS:
        B = _loaded(tag)
        for other in ("Two runs of H-LSTM", "Five runs of H-LSTM", "Two runs of H-LSTM + LightGBM", "Five runs of H-LSTM + LightGBM", TRIPLE):
            rows.append(dict(data=tag, **_cmp(B, HYB, other)))
        for other in ("Two runs of H-LSTM + LightGBM", "Five runs of H-LSTM + LightGBM"):
            rows.append(dict(data=tag, **_cmp(B, TRIPLE, other)))
    _out(rows, "v4_r4_averages.csv")


def _time(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx.get_level_values("time"))


def overlap():
    import paper_v3_paper as P3
    M0, _ = P3.build()
    f0 = M0[P3.TWIN][0].reset_index()
    start = f0[(f0.split == "val") & (f0.fold == 0)].time.min()     # first validation bar of the design runs
    B = _loaded("early_study")
    keep = lambda idx: np.asarray(_time(idx) < start)
    rows = [dict(data="early_study", cut=str(start), **_cmp(B, HYB, o, keep=keep)) for o in (LGBM, NET, HAR)]
    rows += [dict(data="early_study", cut="none", **_cmp(B, HYB, o)) for o in (LGBM, NET, HAR)]
    _out(rows, "v4_r4_overlap.csv")


def bandwidth():
    rows = [dict(data=tag, lag=lag, **_cmp(_loaded(tag), HYB, o, lag=lag)) for tag in SETS for o in (LGBM, NET) for lag in (14, 42, 84)]
    _out(rows, "v4_r4_bandwidth.csv")


def bad_print():
    import paper_v3_paper as P3
    B = _loaded("design")
    t = _time(B[HYB][0].index)
    bar = pd.Timestamp("2025-10-10 20:00", tz=t.tz)
    lo, hi = bar - pd.Timedelta(hours=4), bar + pd.Timedelta(hours=4 * 43)
    keep = lambda idx: np.asarray((_time(idx) < lo) | (_time(idx) > hi))
    rows = [dict(sample="without the bar and its windows", **_cmp(B, HYB, o, keep=keep)) for o in (HAR, LGBM, NET)]
    rows += [dict(sample="all bars", **_cmp(B, HYB, o)) for o in (HAR, LGBM, NET)]
    M0, _ = P3.build()
    f = M0[P3.TWIN][0].reset_index()
    y = f[f.split == "test"].y
    print("test targets:", len(y), "at the floor log(1e-6):", int((y <= np.log(1e-6) + 1e-9).sum()), "smallest target:", float(y.min()))
    _out(rows, "v4_r4_print.csv")


def flat():
    import paper_v2_round2 as R
    import paper_v3_paper as P3
    import paper_v3_runs as RU
    import paper_v3_tables as T
    pattern = [g[2] for g in T.GRID if g[1] == "flat QLSTM / LSTM"][0]
    F = RU.forecasts(("PRMC", "lstm-pool"), [pattern])
    runs = sorted(F)
    print("runs of the single LSTM:", runs)
    M0, _ = P3.build()
    ref = M0[P3.TWIN][0].index
    M = {NET: {k: M0[P3.TWIN][k] for k in runs}, LGBM: {k: M0[P3.LGBM][k] for k in runs}, HYB: {k: M0[P3.TWIN_HYBRID][k] for k in runs},
         "Single LSTM": {k: F[k].reindex(ref) for k in runs}}
    assert all(np.isfinite(f.pred).all() for f in M["Single LSTM"].values())
    M["Single LSTM + LightGBM"] = {k: mix(M["Single LSTM"][k], M[LGBM][k]) for k in runs}
    B = _bars(M, R.train_caps())
    rows = [dict(runs=len(runs), **_cmp(B, a, b)) for a, b in (("Single LSTM", NET), ("Single LSTM + LightGBM", HYB), ("Single LSTM + LightGBM", LGBM),
                                                                ("Single LSTM + LightGBM", "Single LSTM"))]
    _out(rows, "v4_r4_flat.csv")


def gains():
    import itertools
    import paper_v2_round2 as R
    import paper_v3_paper as P3
    import paper_v3_runs as RU
    import paper_v4_round3 as R3
    M0, _ = P3.build()
    ref = M0[P3.TWIN][0].index
    cells = {"network, window": M0[P3.TWIN], "tree, summaries": M0[P3.LGBM], "tree, window": RU.forecasts(("PMW", "lgbmw-pool"), [R3.VIEWS_DIR]),
             "network, summaries": RU.forecasts(("PRMC", "smlp-pool"), [R3.SMLP_RUNS])}
    M = {n: {k: r[k].reindex(ref) for k in SEEDS} for n, r in cells.items()}
    pairs = list(itertools.combinations(cells, 2))
    for a, b in pairs:
        M[f"{a} + {b}"] = {k: mix(M[a][k], M[b][k]) for k in SEEDS}
    B = _bars(M, R.train_caps())
    mse = {n: np.array([x.mse.mean() for x in r]) for n, r in B.items()}
    learner, view = (lambda n: n.split(",")[0]), (lambda n: n.split(",")[1].strip())
    kind = lambda a, b: "input only" if learner(a) == learner(b) else "kind of model only" if view(a) == view(b) else "both"
    G, rows = {}, []
    for a, b in pairs:
        g = 100 * (1 - mse[f"{a} + {b}"] / ((mse[a] + mse[b]) / 2))
        G[(a, b)] = g
        rows.append(dict(kind=kind(a, b), pair=f"{a} + {b}", gain_mean=g.mean(), gain_sd=sd(g), gain_min=g.min(), gain_max=g.max()))
    m = lambda k: np.mean([g for (a, b), g in G.items() if kind(a, b) == k], axis=0)
    d = m("kind of model only") - m("input only")
    half = stats.t.ppf(0.975, len(d) - 1) * sd(d) / np.sqrt(len(d))
    rows.append(dict(kind="difference", pair="kind of model only minus input only (means of two pairs)", gain_mean=d.mean(), gain_sd=sd(d), gain_min=d.mean() - half,
                     gain_max=d.mean() + half, p=float(stats.ttest_1samp(d, 0).pvalue)))
    _out(rows, "v4_r4_gains.csv")


def _out(rows: list, name: str):
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, name), index=False)
    pd.set_option("display.width", 280, "display.max_columns", 40, "display.max_rows", 200)
    print(f"== {name}")
    print(t.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    todo = {"averages": averages, "overlap": overlap, "bandwidth": bandwidth, "print": bad_print, "flat": flat, "gains": gains}
    for cmd in (list(todo) if sys.argv[1] == "all" else sys.argv[1:]):
        todo[cmd]()
