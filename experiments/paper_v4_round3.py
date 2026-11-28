"""
Third round for manuscript v4, after the panel of five simulated reviewers of 2026-11-24
(docs/paper/review/v4_round2/editorial_synthesis.md, points 1 and 3).

Plan (2026-11-24, fixed before any of these models is fitted and before any number below is computed; committed
before the runs). Known before: every table of manuscript v4, of the second round and of the new coins. "The hybrid"
is H-LSTM + LightGBM. Tests and rule: paper_v3_tests (Welch over five runs, Diebold-Mariano over time; better or worse
only if both p < 0.05).

1. VIEWS: does the gain of the hybrid follow the form of the input or the kind of model?
   In the paper the network always reads the window and the tree model always reads the lagged summaries, so the two
   are confounded. Two models cross them, on the coins, folds, seeds 0-4 and protocol of the paper:
     tree on the window   LightGBM with the settings of the paper on the flattened window: the eight price and market
                          features of each of the 42 bars (336 regressors) and the calendar dummies; no lagged
                          averages (`lgbm_window`, results/v4views/lgbmw).
     network on summaries an MLP with two hidden layers and about 20,000 parameters on what the lagged LightGBM-S
                          receives, taken from the window: the features of the last bar, their means over 6 and 42
                          bars, the calendar of the target bar (model `smlp`, configs/paper/v4/views_smlp_pool_4h.yaml).
   With the H-LSTM (network, window) and LightGBM (tree, summaries) these are the four cells of a two-by-two design.
   tables/v4_r3_views.csv: MSE of the four models, and for each of the six plain averages of two of them the MSE, the
   correlation of the forecast errors, the gain G = 1 - MSE(average) / mean of the two member MSEs, the difference to
   the better member, and the two tests against the hybrid.
   Reading, fixed in advance: view-only pairs are (tree on window, tree on summaries) and (network on window, network
   on summaries); learner-only pairs are (network, tree) on the window and (network, tree) on the summaries. The name
   "dual-view" stays in the title only if the mean G of the two view-only pairs is at least as large as the mean G of
   the two learner-only pairs. Otherwise the paper is retitled around the averaging of models of different kinds, and
   the two views are described as the cheapest way to obtain them, not as the source of the gain. Reported whatever
   it is.

2. EARLIER PERIOD: the frozen system on a period for which no forecast was ever evaluated.
   Four test blocks of 135 days that end where the first test block of the paper begins (30 October 2023 to
   22 April 2025; validation 180 days, training everything before), for two sets of coins: the 15 coins of the study
   and the 15 coins of paper_v4_newcoins. Exchange data only, samples from 2020-01-01 or the listing
   (`exchange_only`); for the coins of the study these samples are longer than those of the paper, which a vendor
   feed limits. For the coins of the study the period was training and validation data of every earlier run; for the
   new coins it was never used. Models: hl_gattn, hq_gattn and the Transformer, seeds 0-4
   (configs/paper/v4/early_{study,new}_pool_4h.yaml); the lagged LightGBM-S, HAR and HAR-S x (`early_save`).
   Evaluation: paper_v4_newcoins.evaluate, unchanged (tables/v4_early_{study,new}.csv and *_limits.csv).
   PRIMARY, for each set of coins: the result counts as confirmed on the earlier period if the hybrid has a lower MSE
   than LightGBM and than the H-LSTM by the rule. The other comparisons are secondary. Reported whatever it is.

3. BLOCK AFTER THE SAMPLE, to be run once, on the day before submission: one fold on all four-hour bars after
   2026-10-14 16:00 UTC up to the last complete bar at the download, for both sets of coins, with the models, seeds
   and evaluation of point 2. The block will hold about 50 days, too few to establish a difference of 1-2%; the ratio
   to LightGBM and its interval are reported as they are, and the abstract states the outcome.

4. FURTHER ANALYSES asked for by the panel (added 2026-11-24 02:30 and committed before they were computed; the
   outcome of point 1 was known by then, that of point 2 was not):
     cost      tables/v4_r3_cost.csv: averages at a higher cost against the hybrid and against the three-member
               average with the Transformer: two runs of the H-LSTM with LightGBM (thirds), the five-run average of
               the H-LSTM with LightGBM (halves).
     variance  tables/v4_r3_variance.csv: the hybrid formed on the variance scale (mean of the two calibrated
               variance forecasts instead of the mean of the log forecasts): QLIKE against the range (cap and level
               constant of the paper) against LightGBM and against the hybrid formed in logs.
     limits    tables/v4_r3_limits.csv: the range limits of paper_v3_paper.limits at 95%, 99% and 99.5% for HAR,
               HAR-S x, LightGBM, the H-LSTM, the Transformer, the hybrid and the three-member average: exceedance
               rate, tick loss, mean and largest excess of the range over the limit; the hybrid against each on the
               tick loss. And LightGBM fitted to the quantile directly (objective "quantile", otherwise the settings
               and regressors of the paper, five seeds) at 95% and 99%, whose forecast is the limit.
     linear    tables/v4_r3_linear.csv: pooled linear benchmarks, fitted like the pooled LightGBM (per-coin
               standardisation, one fit on all coins): HAR, HAR with calendar dummies, and a linear regression on the
               35 regressors of LightGBM; and the conditional autoregressive range model (CARR) per coin. The hybrid
               against each; the average of the H-LSTM with the pooled linear model on the 35 regressors.

    .venv/bin/python -W ignore experiments/paper_v4_round3.py lgbm_window
    .venv/bin/python -W ignore experiments/paper_v4_round3.py early_save [study|new]
    .venv/bin/python -W ignore experiments/paper_v4_round3.py views           tables/v4_r3_views.csv
    .venv/bin/python -W ignore experiments/paper_v4_round3.py early [study|new]
    .venv/bin/python -W ignore experiments/paper_v4_round3.py fit_extra        quantile LightGBM, pooled linear models, CARR
    .venv/bin/python -W ignore experiments/paper_v4_round3.py extra            the four tables of point 4
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

CSV, SEEDS, FREQ = "docs/paper/manuscript_v2/tables", (0, 1, 2, 3, 4), "4h"
VIEWS_DIR, SMLP_RUNS = "results/v4views/lgbmw", "results/v4views/out/v4_views_smlp_pool_4h_*"
EARLY_END = "2025-04-22 16:00"
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def lgbm_window():
    """LightGBM of the paper on the flattened window instead of the lagged averages."""
    import lightgbm  # noqa: F401  (its OpenMP runtime first, as in the runner)
    from paper_round7_checks import _frames
    from paper_v4_bench import _save
    from src.paper.baselines import predict_baseline_pooled
    from src.paper.data import GROUPS
    fr, win = _frames(FREQ)
    dss = list(fr)
    cols = GROUPS["price_r"] + GROUPS["market"]
    Xw = [np.column_stack([fr[ds][0].frame[cols].shift(k).to_numpy(np.float32) for k in range(win)]) for ds in dss]
    for seed in SEEDS:
        for k in range(len(fr[dss[0]][1])):
            if all(os.path.exists(os.path.join(VIEWS_DIR, "preds", f"{ds}__range__PMW__lgbmw-pool__s{seed}__f{k}.npz")) for ds in dss):
                continue
            res = predict_baseline_pooled("winlgbms", [fr[ds][0].frame for ds in dss], "range", [fr[ds][1][k] for ds in dss], win, FREQ, Xw, seed=seed)
            for ds, r in zip(dss, res):
                _save(VIEWS_DIR, "lgbmw-pool", ds, "PMW", seed, k, fr[ds][0].frame, r)
            print("window LightGBM seed", seed, "fold", k, flush=True)


def views():
    import paper_v2_round2 as R
    import paper_v3_paper as P3
    import paper_v3_runs as RU
    from paper_v3_paper import LGBM, compare
    from paper_v4_paper import _losses
    M0, _ = P3.build()
    HYB = P3.TWIN_HYBRID
    ref = M0[P3.TWIN][0].index
    y = M0[P3.TWIN][0].y.to_numpy()
    test = ref.get_level_values("split").to_numpy() == "test"
    frame = lambda pred: pd.DataFrame({"y": y, "pred": np.asarray(pred)}, index=ref)
    cells = {"network, window (H-LSTM)": M0[P3.TWIN], "tree, summaries (LightGBM)": M0[LGBM],
             "tree, window": RU.forecasts(("PMW", "lgbmw-pool"), [VIEWS_DIR]), "network, summaries": RU.forecasts(("PRMC", "smlp-pool"), [SMLP_RUNS])}
    P = {n: {k: r[k].reindex(ref).pred.to_numpy() for k in SEEDS} for n, r in cells.items()}
    assert all(np.isfinite(v).all() for r in P.values() for v in r.values())
    learner, view = (lambda n: n.split(",")[0]), (lambda n: n.split(",")[1].split()[0])
    kind = lambda a, b: "view only" if learner(a) == learner(b) else "learner only" if view(a) == view(b) else "both"
    allm = {n: {k: frame(P[n][k]) for k in SEEDS} for n in cells}
    pairs = list(itertools.combinations(cells, 2))
    for a, b in pairs:
        allm[f"{a} + {b}"] = {k: frame((P[a][k] + P[b][k]) / 2) for k in SEEDS}
    allm[HYB] = M0[HYB]
    per_run, per_bar = _losses(allm, R.train_caps())
    rows = [dict(kind="single", name=n, mse_mean=per_run[n].mse.mean(), mse_sd=sd(per_run[n].mse), qlike_mean=per_run[n].qlike.mean()) for n in cells]
    for a, b in pairs:
        name = f"{a} + {b}"
        corr = float(np.mean([np.corrcoef(P[a][k][test] - y[test], P[b][k][test] - y[test])[0, 1] for k in SEEDS]))
        ma, mb, mp = per_run[a].mse.mean(), per_run[b].mse.mean(), per_run[name].mse.mean()
        r = dict(kind=kind(a, b), name=name, mse_mean=mp, mse_sd=sd(per_run[name].mse), qlike_mean=per_run[name].qlike.mean(), corr=corr,
                 gain_pct=100 * (1 - mp / ((ma + mb) / 2)), vs_better_pct=100 * (mp / min(ma, mb) - 1))
        r.update({"hyb_" + c: v for c, v in compare(per_run, per_bar, name, HYB).items() if c.startswith("mse_")})
        rows.append(r)
    t = pd.DataFrame(rows)
    assert np.isclose(t.loc[t.name == "network, window (H-LSTM) + tree, summaries (LightGBM)", "mse_mean"].iloc[0], per_run[HYB].mse.mean(), rtol=1e-9)
    t.to_csv(os.path.join(CSV, "v4_r3_views.csv"), index=False)
    pd.set_option("display.width", 260, "display.max_columns", 30)
    print(t[["kind", "name", "mse_mean", "mse_sd", "corr", "gain_pct", "vs_better_pct", "hyb_mse_diff_pct", "hyb_mse_p_runs", "hyb_mse_p_time", "hyb_mse_verdict"]].round(4).to_string(index=False))
    g = t[t.kind != "single"].groupby("kind").gain_pct.mean()
    print(g.round(3).to_string())
    print("READING:", "the view-only gain is at least the learner-only gain: 'dual-view' stays" if g["view only"] >= g["learner only"]
          else "the learner-only gain is larger than the view-only gain: retitle around the averaging of models of different kinds")


EXTRA_DIR = "results/v4views"
QUANTILES = (0.95, 0.99)


def fit_extra():
    """Quantile LightGBM, the pooled linear benchmarks and CARR (point 4 of the plan)."""
    import lightgbm  # noqa: F401
    import src.paper.baselines as B
    from paper_round7_checks import _cfg, _frames
    from paper_v4_bench import _save
    fr, win = _frames(FREQ)
    dss = list(fr)
    Xs = [fr[ds][0].X(_cfg(f"lgbm_{FREQ}")["groupsets"]["PM"]) for ds in dss]
    Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in B.HAR_LAGS[FREQ][1:]]) for X in Xs]
    frames, n_folds = [fr[ds][0].frame for ds in dss], len(fr[dss[0]][1])
    for k in range(n_folds):
        folds = [fr[ds][1][k] for ds in dss]
        for name, label in (("har", "har-pool"), ("hars", "hars-pool"), ("harxs", "linear-pool")):
            for ds, r in zip(dss, B.predict_baseline_pooled(name, frames, "range", folds, win, FREQ, Xl)):
                _save(f"{EXTRA_DIR}/linear", label, ds, "PM", 0, k, fr[ds][0].frame, r)
        for ds in dss:
            _save(f"{EXTRA_DIR}/linear", "carr", ds, "P", 0, k, fr[ds][0].frame, B.predict_baseline("carr", fr[ds][0].frame, "range", fr[ds][1][k], win, FREQ))
        print("linear and CARR, fold", k, flush=True)
    default = dict(B.LGBM_PARAMS)
    for q in QUANTILES:
        B.LGBM_PARAMS = {**default, "objective": "quantile", "alpha": q}
        for seed in SEEDS:
            for k in range(n_folds):
                res = B.predict_baseline_pooled("lgbms", frames, "range", [fr[ds][1][k] for ds in dss], win, FREQ, Xl, seed=seed)
                for ds, r in zip(dss, res):
                    _save(f"{EXTRA_DIR}/lgbmq{int(round(100 * q))}", "lgbmq-pool", ds, "PM", seed, k, fr[ds][0].frame, r)
            print("quantile LightGBM", q, "seed", seed, flush=True)
    B.LGBM_PARAMS = default


def extra():
    import paper_v2_round2 as R
    import paper_v3_paper as P3
    import paper_v3_runs as RU
    import paper_v4_round2 as R2
    from paper_v3_paper import LGBM, compare
    from paper_v4_paper import _losses
    get = RU.forecasts
    M, _ = P3.build()
    HYB, NET, TRF = P3.TWIN_HYBRID, P3.TWIN, "Transformer"
    ref = M[NET][0].index
    y = M[NET][0].y.to_numpy()
    frame = lambda pred: pd.DataFrame({"y": y, "pred": np.asarray(pred)}, index=ref)
    P = {n: {k: M[n][k].reindex(ref).pred.to_numpy() for k in SEEDS} for n in (NET, LGBM, TRF)}
    caps = R.train_caps()
    TRIPLE = "H-LSTM + LightGBM + Transformer"
    M[TRIPLE] = {k: frame((P[NET][k] + P[LGBM][k] + P[TRF][k]) / 3) for k in SEEDS}
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 100)

    # cost
    mean5 = np.mean([P[NET][k] for k in SEEDS], axis=0)
    X = {HYB: M[HYB], TRIPLE: M[TRIPLE],
         "Two runs of H-LSTM + LightGBM": {k: frame((P[NET][k] + P[NET][(k + 1) % 5] + P[LGBM][k]) / 3) for k in SEEDS},
         "Five runs of H-LSTM + LightGBM": {k: frame((mean5 + P[LGBM][k]) / 2) for k in SEEDS}}
    per_run, per_bar = _losses(X, caps)
    rows = [dict(name=n, mse_mean=per_run[n].mse.mean(), mse_sd=sd(per_run[n].mse), **{"hyb_" + c: v for c, v in compare(per_run, per_bar, n, HYB).items() if c.startswith("mse_")},
                 **{"triple_" + c: v for c, v in compare(per_run, per_bar, n, TRIPLE).items() if c.startswith("mse_")}) for n in list(X)[2:]]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v4_r3_cost.csv"), index=False)
    print(t.round(4).to_string(index=False))

    # variance: the hybrid formed on the variance scale
    cal = lambda f: R._calibrated(f.reset_index().assign(groupset="g", model="m"), None, caps=caps).set_index(["time", "dataset"]).sort_index()
    runs, bars = [], []
    for k in SEEDS:
        a, b = cal(M[NET][k]), cal(M[LGBM][k])
        h = (a.h + b.h.reindex(a.index)) / 2
        ql = a.s2 / h - np.log(a.s2 / h) - 1
        runs.append(float(ql.mean())); bars.append(ql)
    v_run, v_bar = np.array(runs), sum(bars) / len(bars)
    pr, pb = _losses({HYB: M[HYB], LGBM: M[LGBM]}, caps)
    rows = [dict(name="hybrid on the variance scale", qlike_mean=v_run.mean(), qlike_sd=sd(v_run), other=o, other_qlike=pr[o].qlike.mean(),
                 **R2._two_tests(v_run, pr[o].qlike.to_numpy(), v_bar, pb[o].qlike)) for o in (LGBM, HYB)]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v4_r3_variance.csv"), index=False)
    print(t.round(4).to_string(index=False))

    # limits at three levels, and LightGBM fitted to the quantile
    names = ["HAR", "HAR-S x", LGBM, NET, TRF, HYB, TRIPLE]
    rows = []
    for level in (0.95, 0.99, 0.995):
        L = R2._limits(M, names, rate=1 - level)
        stat = {}
        for n in names:
            runs = L[n]
            excess = [np.exp((x.y - x.pred - x.c)[x.hit == 1]) - 1 for x in runs]
            bars = [x.set_index(["time", "dataset"]).tick.sort_index() for x in runs]
            stat[n] = dict(tick=np.array([x.tick.mean() for x in runs]), bar=sum(bars) / len(bars), exceed=np.mean([x.hit.mean() for x in runs]),
                           excess_mean=np.mean([e.mean() for e in excess]), excess_max=np.max([e.max() for e in excess]))
        if level in QUANTILES:
            q = get(("PM", "lgbmq-pool"), [f"{EXTRA_DIR}/lgbmq{int(round(100 * level))}"])
            runs = []
            for k in SEEDS:
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
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v4_r3_limits.csv"), index=False)
    print(t.round(4).to_string(index=False))

    # pooled linear benchmarks and CARR
    lin = {"HAR, pooled": {0: get(("PM", "har-pool"), [f"{EXTRA_DIR}/linear"])[0].reindex(ref)}, "HAR-S, pooled": {0: get(("PM", "hars-pool"), [f"{EXTRA_DIR}/linear"])[0].reindex(ref)},
           "Linear on the 35 regressors, pooled": {0: get(("PM", "linear-pool"), [f"{EXTRA_DIR}/linear"])[0].reindex(ref)}, "CARR": {0: get(("P", "carr"), [f"{EXTRA_DIR}/linear"])[0].reindex(ref)}}
    lin["H-LSTM + linear on the 35 regressors"] = {k: frame((P[NET][k] + lin["Linear on the 35 regressors, pooled"][0].pred.to_numpy()) / 2) for k in SEEDS}
    per_run, per_bar = _losses({HYB: M[HYB], LGBM: M[LGBM], "HAR": M["HAR"], **lin}, caps)
    rows = [dict(model=n, mse_mean=per_run[n].mse.mean(), mse_sd=sd(per_run[n].mse), mse_har=per_run[n].mse.mean() / per_run["HAR"].mse.mean(), qlike_mean=per_run[n].qlike.mean(),
                 **{"hyb_" + c: v for c, v in compare(per_run, per_bar, HYB, n).items() if c.startswith("mse_")},
                 **{"lgbm_" + c: v for c, v in compare(per_run, per_bar, LGBM, n).items() if c.startswith("mse_")}) for n in lin]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v4_r3_linear.csv"), index=False)
    print(t.round(4).to_string(index=False))


def _early(tag: str):
    """paper_v4_newcoins with the earlier folds and one of the two sets of coins."""
    import paper_v4_newcoins as N
    coins = N.OLD if tag == "study" else json.load(open(N.UNIVERSE))["chosen"]
    N.chosen = lambda: coins
    N.TAG, N.FOLDS = f"early_{tag}", dict(n=4, test_days=135, val_days=180, end=EARLY_END)
    N.BENCH, N.OUT_RUNS = f"results/v4early/bench_{tag}", f"results/v4early/out/v4_early_{tag}_pool_4h_*"
    return N


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "lgbm_window":
        lgbm_window()
    elif cmd == "views":
        views()
    elif cmd == "early_save":
        _early(rest[0]).save()
    elif cmd == "early":
        _early(rest[0]).evaluate()
    elif cmd == "fit_extra":
        fit_extra()
    elif cmd == "extra":
        extra()
