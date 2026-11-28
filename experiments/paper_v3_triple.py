"""
Three models in one hybrid: HAR, the lagged LightGBM-S and the H-QLSTM (asked for by the advisor, 2026-11-18:
three models, with stacking and with blending, as 'HAR + LightGBM + HQLSTM' and 'HAR + HQLSTM + LightGBM').
In blending and stacking the order of the parts does not matter; in a cascade it does, so the two orders named
there are run as cascades. Fixed before anything of this file was computed. Known before: the plain average and the
linear stack of candidate A, LightGBM and the seasonal HAR on three-seed averages, 0.821 and 0.815 x HAR
(tables/v3_hybrid_har.csv), and every two-model hybrid (tables/v3_runs.csv).

  parts       H   HAR (three lags of the log range, OLS per coin), one fit
              G   pooled LightGBM-S with lagged exchange features, the benchmark of the paper
              N   the H-QLSTM, candidate A (hq_gattn on price with range, market, calendar); R its real twin
  run         k = 0..4 as in experiments/paper_v3_runs.py: run k of a combination uses run k of N and of G.
  blend3_avg  (H + G + N) / 3
  blend3_w    a H + b G + c N with a, b, c >= 0 and a + b + c = 1 on a grid of step 0.01: least MSE on the
              validation block of the fold, one set of weights for all coins
  stack3_lin  OLS of the target on (1, H, G, N), fitted on the validation block of the fold
  ..._sx      the same three with the seasonal HAR (HAR-S x) in the place of HAR
  blend3_avg_real   (H + G + R) / 3: does the algebra matter inside the three-way average?
  cascade H>G>N     stage 1 HAR; stage 2 the lagged LightGBM-S (regressors and parameters of the benchmark, three
              seeds averaged) fitted on y - HAR; stage 3 the H-QLSTM trained on what the two leave, over the
              whole training block. On the training rows stages 1 and 2 are out of fold over five blocks of
              time (experiments/paper_v3_cascade.py offsets harlgbm). configs/paper/v3/a_cascade_har_lgbm_pool_4h.yaml,
              seeds 0-4, with the real twin as the third stage as well. By-product: the two stages H>G alone.
  cascade H>N>G     stage 1 HAR; stage 2 the H-QLSTM trained on y - HAR (the cascade of a_cascade_har_*_pool_4h,
              five seeds exist); stage 3 the lagged LightGBM-S (same regressors and parameters, seed k) fitted on
              y - (HAR + network k), early stopping on the validation block. On the training rows the forecast
              of stages 1-2 is out of fold: network k is trained five more times, each time without one of the
              five blocks, and forecasts that block (configs a_oof_har_b<j>_pool_4h.yaml, runner key `oof`;
              `third` below).
  reported    as in paper_v3_runs: mean and standard deviation over the five runs of MSE and QLIKE on the four
              test blocks, in the units of the loss and relative to HAR. For the average of the five runs'
              forecasts: the ratio of the MSE to that of the plain average of N and G, the best hybrid so far,
              with the 90% interval and the cross-coin DM test. A three-model hybrid counts as better than the
              two-model one only if that ratio is below one with p < 0.05. Every row is reported.
  not done    the 33 days after the sample: the cascades would need further training there, and three seeds exist.

    .venv/bin/python -W ignore experiments/paper_v3_triple.py third    stage 3 of H>N>G (results/v3runs)
    .venv/bin/python -W ignore experiments/paper_v3_triple.py          tables/v3_triple.csv, v3_triple_weights.csv
    .venv/bin/python -W ignore experiments/paper_v3_triple.py fig      review/advisor_2026-11-17/candidate_a/three_model_table.png
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_runs as RU  # noqa: E402

OUT = RU.OUT
FREQ, WINDOW, K_BLOCKS, RUNS, V3 = "4h", 42, 5, RU.RUNS, RU.V3
NET, TWIN, LGBM, HAR = ("PRMC", "hq_gattn-pool"), ("PRMC", "hl_gattn-pool"), ("PM", "lgbmsl-pool"), ("P", "har")
HN_RUNS = [V3.format("cascade_har_pool_4h"), V3.format("cascade_har_seeds_pool_4h")]       # H>N, validation and test rows
HGN_RUNS = [V3.format("cascade_har_lgbm_pool_4h")]
OOF_RUN = V3.format("oof_har_b{}_pool_4h")                                                  # H>N on block j of the training rows
HG_OFFSETS = "data/offsets/v3_cascade_har_lgbm_4h"
HNG_DIR, HNG = "results/v3runs/v3_cascade_har_net_lgbm_4h", ("PRMC", "har-hq_gattn-lgbmsl")
BEST = "Blending: H-QLSTM + LightGBM"
ORDER = ["HAR", "HAR-S x", "LightGBM-S, lagged", "H-QLSTM (candidate A)", BEST,
         "Blending: HAR + LightGBM + H-QLSTM, average", "Blending: HAR + LightGBM + H-QLSTM, weighted",
         "Stacking: HAR + LightGBM + H-QLSTM", "Blending: HAR + LightGBM + H-LSTM, average",
         "Blending: HAR-S x + LightGBM + H-QLSTM, average", "Blending: HAR-S x + LightGBM + H-QLSTM, weighted",
         "Stacking: HAR-S x + LightGBM + H-QLSTM",
         "Cascade: HAR, then LightGBM", "Cascade: HAR, then LightGBM, then H-QLSTM", "Cascade: HAR, then LightGBM, then H-LSTM",
         "Cascade: HAR, then H-QLSTM", "Cascade: HAR, then H-QLSTM, then LightGBM"]


def _ns(t) -> np.ndarray:
    """Epoch nanoseconds of a time index or column, with or without a time zone."""
    t = pd.DatetimeIndex(t)
    return (t if t.tz is None else t.tz_convert(None)).as_unit("ns").asi8


def _cells(patterns: list[str], key: tuple, splits=("val", "test")) -> pd.DataFrame:
    """Forecasts of the cells of some runs, long: dataset, seed, fold, split, time in epoch ns, pred."""
    parts = []
    for pattern in patterns:
        dirs = glob.glob(pattern)
        assert dirs, f"no run matches {pattern}"
        for directory in dirs:
            for path in glob.glob(os.path.join(directory, "preds", f"*__range__{key[0]}__{key[1]}__s*__f*.npz")):
                ds, _, _, _, seed, fold = os.path.basename(path)[:-4].split("__")
                z = np.load(path)
                for split in splits:
                    parts.append(pd.DataFrame({"dataset": ds, "seed": int(seed[1:]), "fold": int(fold[1:]), "split": split,
                                               "t": z[f"{split}_time"].astype("int64"), "pred": z[f"{split}_pred"].astype(float)}))
    return pd.concat(parts, ignore_index=True)


def third(seeds=RUNS):
    """Stage 3 of the cascade HAR -> network -> LightGBM: for every fold and run, the lagged LightGBM-S fitted on
    what HAR and the network of that run leave. Training rows: every row of the training block where a window
    ends, with the out-of-fold forecast of stages 1-2; early stopping on the validation block. Writes validation
    and test forecasts of the three stages in the format of the runner."""
    import lightgbm  # noqa: F401  (its OpenMP runtime first, as in the runner)
    from paper_round7_checks import _frames
    from src.paper.baselines import HAR_LAGS, _design, _lgbm
    frames, _ = _frames(FREQ)
    full = _cells(HN_RUNS, NET).set_index(["dataset", "seed", "fold"]).sort_index()
    oof = pd.concat([_cells([OOF_RUN.format(j)], NET, splits=("test",)) for j in range(K_BLOCKS)], ignore_index=True)
    assert not oof.duplicated(["dataset", "seed", "fold", "t"]).any(), "a training row is forecast by two blocks"
    oof = oof.set_index(["dataset", "seed", "fold"]).sort_index()
    os.makedirs(os.path.join(HNG_DIR, "preds"), exist_ok=True)
    json.dump({"config": {"name": "v3_cascade_har_net_lgbm_4h", "note": "paper_v3_triple.third"}}, open(os.path.join(HNG_DIR, "config.json"), "w"), indent=1)
    n_folds = len(next(iter(frames.values()))[1])
    for k in range(n_folds):
        coins = []
        for ds, (d, folds) in frames.items():
            X = d.X(["price", "market"])
            Xl = np.column_stack([X] + [pd.DataFrame(X).rolling(lag).mean().to_numpy() for lag in HAR_LAGS[FREQ][1:]])
            t = _ns(d.frame.index)
            coins.append(dict(ds=ds, D=_design("lgbms", d.frame, FREQ, Xl), y=d.frame["y_range"].to_numpy(float), t=t,
                              row=pd.Series(np.arange(len(t)), index=t), fold=folds[k]))
        for seed in seeds:
            path = lambda ds: os.path.join(HNG_DIR, "preds", f"{ds}__range__{HNG[0]}__{HNG[1]}__s{seed}__f{k}.npz")
            if all(os.path.exists(path(c["ds"])) for c in coins):
                continue
            A, b, Xva, zva, per = [], [], [], [], []
            for c in coins:
                sel = lambda r: r[r >= WINDOW - 1]
                tr, va, te = (sel(c["fold"][name]) for name in ("train", "val", "test"))
                p = np.full(len(c["y"]), np.nan)                        # forecast of stages 1-2 on every row that has one
                for src in (oof, full):
                    g = src.loc[(c["ds"], seed, k)]
                    p[c["row"].loc[g.t.to_numpy()].to_numpy()] = g.pred.to_numpy()
                assert np.isfinite(p[np.concatenate([tr, va, te])]).all(), f"{c['ds']} fold {k} seed {seed}: a row has no forecast of stages 1-2"
                fit = tr[~np.isnan(c["D"][tr]).any(1)]
                mx, sx = c["D"][fit].mean(0), c["D"][fit].std(0)
                sx[sx < 1e-8] = 1.0
                r = c["y"] - p
                my, sy = r[fit].mean(), r[fit].std()
                Dz = (c["D"] - mx) / sx
                A.append(Dz[fit]); b.append((r[fit] - my) / sy); Xva.append(Dz[va]); zva.append((r[va] - my) / sy)
                per.append((c, p, Dz, my, sy, va, te))
            m = _lgbm(np.vstack(A), np.concatenate(b), np.vstack(Xva), np.concatenate(zva), seed=seed)
            for c, p, Dz, my, sy, va, te in per:
                out = {s: p[rows] + my + sy * m.predict(Dz[rows]) for s, rows in (("val", va), ("test", te))}
                np.savez(path(c["ds"]), test_time=c["t"][te], test_y=c["y"][te].astype("float32"), test_pred=out["test"].astype("float32"),
                         val_time=c["t"][va], val_y=c["y"][va].astype("float32"), val_pred=out["val"].astype("float32"),
                         meta=np.array(json.dumps(dict(seed=seed, fold=k, model=HNG[1], trees=int(m.best_iteration_ or 0)))))
            print(f"fold {k} seed {seed}: {m.best_iteration_} trees", flush=True)


GRID = np.array([(i, j, 100 - i - j) for i in range(101) for j in range(101 - i)]) / 100.0     # weights on (h, g, n)


def blend_w(x: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Weighted average of three forecasts (columns h, g, n of x, with y): per fold the weights of GRID with the
    least MSE on the validation block, applied to both blocks."""
    out, rec = x[["y"]].assign(pred=np.nan), []
    for fold, f in x.groupby(level="fold"):
        va = f[f.index.get_level_values("split") == "val"]
        e = va[["h", "g", "n"]].to_numpy() - va[["y"]].to_numpy()
        w = GRID[np.argmin(np.einsum("ij,jk,ik->i", GRID, e.T @ e / len(e), GRID))]
        out.loc[f.index, "pred"] = f[["h", "g", "n"]].to_numpy() @ w
        rec.append(dict(fold=int(fold), w_har=w[0], w_lgbm=w[1], w_network=w[2]))
    return out, rec


def stack(x: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Linear stack of three forecasts: per fold an OLS of y on (1, h, g, n) on the validation block, applied to both blocks."""
    out, rec = x[["y"]].assign(pred=np.nan), []
    for fold, f in x.groupby(level="fold"):
        va = f[f.index.get_level_values("split") == "val"]
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(va)), va[["h", "g", "n"]]]), va.y.to_numpy(), rcond=None)[0]
        out.loc[f.index, "pred"] = beta[0] + f[["h", "g", "n"]].to_numpy() @ beta[1:]
        rec.append(dict(fold=int(fold), intercept=beta[0], b_har=beta[1], b_lgbm=beta[2], b_network=beta[3]))
    return out, rec


def two_stages(ref: pd.DataFrame) -> pd.DataFrame:
    """HAR, then LightGBM on its residual: the saved forecast of the two stages on the validation and test rows of `ref`."""
    out = ref.copy()
    t = _ns(ref.index.get_level_values("time"))
    ds, fold = ref.index.get_level_values("dataset"), ref.index.get_level_values("fold")
    pred = np.full(len(ref), np.nan)
    for (d, k) in sorted(set(zip(ds, fold))):
        z = np.load(os.path.join(HG_OFFSETS, f"{d}__f{k}.npz"))
        m = np.asarray((ds == d) & (fold == k))
        pred[m] = pd.Series(z["offset"].astype(float), index=z["time"]).reindex(t[m]).to_numpy()
    assert np.isfinite(pred).all()
    return out.assign(pred=pred)


def main():
    import paper_v2_round2 as R
    caps = R.train_caps()
    get = lambda key, pats: RU.forecasts(key, pats)
    A, T, G = get(NET, RU.A_RUNS), get(TWIN, RU.A_RUNS), get(*RU.SEEDED["LightGBM-S, lagged"])
    har, harsx = get(*RU.FIXED["HAR"])[0], get(*RU.FIXED["HAR-S x"])[0]

    def wide(k: int, net: dict, third_: pd.DataFrame) -> pd.DataFrame:
        a = net[k]
        x = pd.DataFrame({"y": a.y, "h": third_.reindex(a.index).pred, "g": G[k].reindex(a.index).pred, "n": a.pred})
        assert x.notna().all().all()
        return x

    runs, weights = {}, []
    both = [k for k in RUNS if k in A and k in G]
    runs["LightGBM-S, lagged"], runs["H-QLSTM (candidate A)"] = G, A
    runs[BEST] = {k: A[k].assign(pred=(A[k].pred + G[k].reindex(A[k].index).pred) / 2) for k in both}
    for label, third_ in (("HAR", har), ("HAR-S x", harsx)):
        avg, wgt, stk = {}, {}, {}
        for k in both:
            x = wide(k, A, third_)
            avg[k] = x[["y"]].assign(pred=x[["h", "g", "n"]].mean(1))
            wgt[k], rec_w = blend_w(x)
            stk[k], rec_s = stack(x)
            weights += [dict(third=label, run=k, **a, **{key: v for key, v in b.items() if key != "fold"}) for a, b in zip(rec_w, rec_s)]
        runs[f"Blending: {label} + LightGBM + H-QLSTM, average"] = avg
        runs[f"Blending: {label} + LightGBM + H-QLSTM, weighted"] = wgt
        runs[f"Stacking: {label} + LightGBM + H-QLSTM"] = stk
    runs["Blending: HAR + LightGBM + H-LSTM, average"] = {k: (lambda x: x[["y"]].assign(pred=x[["h", "g", "n"]].mean(1)))(wide(k, T, har))
                                                         for k in RUNS if k in T and k in G}
    runs["Cascade: HAR, then H-QLSTM"] = get(NET, HN_RUNS)
    runs["Cascade: HAR, then LightGBM, then H-QLSTM"] = get(NET, HGN_RUNS) if glob.glob(HGN_RUNS[0]) else {}
    runs["Cascade: HAR, then LightGBM, then H-LSTM"] = get(TWIN, HGN_RUNS) if glob.glob(HGN_RUNS[0]) else {}
    runs["Cascade: HAR, then H-QLSTM, then LightGBM"] = get(HNG, [HNG_DIR])
    fixed = {"HAR": har, "HAR-S x": harsx}
    if len(glob.glob(os.path.join(HG_OFFSETS, "*__f*.npz"))) == 60:                 # 15 coins x 4 folds
        fixed["Cascade: HAR, then LightGBM"] = two_stages(har)

    mse_har, ql_har = RU.losses(har, caps)
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    rows, ens = [], {}
    for name in ORDER:
        if name in fixed:
            per, have, ens[name] = np.array([RU.losses(fixed[name], caps)]), [0], fixed[name]
        else:
            have = [k for k in RUNS if k in runs.get(name, {})]
            if not have:
                print("not available:", name)
                continue
            for k in have:
                assert np.allclose(runs[name][k].y, har.reindex(runs[name][k].index).y, atol=1e-4), f"{name}: targets differ from those of HAR"
            per = np.array([RU.losses(runs[name][k], caps) for k in have])
            first = runs[name][have[0]]
            ens[name] = first.assign(pred=np.mean([runs[name][k].reindex(first.index).pred for k in have], axis=0))
        mse_avg, ql_avg = RU.losses(ens[name], caps)
        rows.append(dict(model=name, runs=len(have) if name not in fixed else 1, mse_mean=per[:, 0].mean(), mse_sd=sd(per[:, 0]),
                         mse_har_mean=(per[:, 0] / mse_har).mean(), mse_har_sd=sd(per[:, 0] / mse_har),
                         qlike_mean=per[:, 1].mean(), qlike_sd=sd(per[:, 1]),
                         qlike_har_mean=(per[:, 1] / ql_har).mean(), qlike_har_sd=sd(per[:, 1] / ql_har),
                         mse_har_avg=mse_avg / mse_har, qlike_har_avg=ql_avg / ql_har))
    t = pd.DataFrame(rows)

    def long(name: str) -> pd.DataFrame:
        e = ens[name][ens[name].index.get_level_values("split") == "test"].reset_index()
        key = HAR if name == "HAR" else ("HY", name)
        return pd.DataFrame({"dataset": e.dataset, "task": "range", "groupset": key[0], "model": key[1], "seed": 0,
                             "fold": e.fold, "time": e.time, "y": e.y, "pred": e.pred})

    st = R.Store([long(name) for name in ens])
    for other, short in ((BEST, "best"), ("LightGBM-S, lagged", "lgbm")):          # averages of the runs' forecasts against the two references
        for i, name in enumerate(t.model):
            if name in ("HAR", other):
                continue
            v = st.pair(("HY", name), ("HY", other))
            for col, val in ((f"vs_{short}", v["ratio"]), (f"vs_{short}_lo", v["ci_lo"]), (f"vs_{short}_hi", v["ci_hi"]),
                             (f"vs_{short}_better", v["better"]), (f"p_{short}", v["p"])):
                t.loc[i, col] = val
    t.to_csv(os.path.join(OUT, "v3_triple.csv"), index=False)
    w = pd.DataFrame(weights)
    w.groupby(["third", "fold"]).mean(numeric_only=True).drop(columns="run").reset_index().to_csv(os.path.join(OUT, "v3_triple_weights.csv"), index=False)
    pm = lambda m, s: f"{m:.4f}" + (f" ± {s:.4f}" if np.isfinite(s) else "")
    show = pd.DataFrame({"model": t.model, "runs": t.runs, "MSE": [pm(m, s) for m, s in zip(t.mse_mean, t.mse_sd)],
                         "MSE / HAR": [pm(m, s) for m, s in zip(t.mse_har_mean, t.mse_har_sd)],
                         "QLIKE": [pm(m, s) for m, s in zip(t.qlike_mean, t.qlike_sd)],
                         "avg: MSE / HAR": t.mse_har_avg.round(4), "vs best": t.get("vs_best", pd.Series(dtype=float)).round(4),
                         "p": t.get("p_best", pd.Series(dtype=float)).round(4), "coins": t.get("vs_best_better", pd.Series(dtype=float))})
    pd.set_option("display.width", 260, "display.max_columns", 20)
    print(show.to_string(index=False))
    print(w.groupby(["third", "fold"]).mean(numeric_only=True).drop(columns="run").round(3).to_string())


FIG = "docs/paper/review/advisor_2026-11-17/candidate_a/three_model_table.png"
FIG_GROUPS = [   # (heading, [(label, row of tables/v3_triple.csv)])
    ("Single models and the best two-model hybrid", [("HAR", "HAR"), ("LightGBM-S, lagged features", "LightGBM-S, lagged"),
                                                    ("H-QLSTM (candidate A)", "H-QLSTM (candidate A)"),
                                                    ("Blending: average of H-QLSTM and LightGBM", BEST)]),
    ("Three models, blending and stacking (the order of the models does not matter)",
     [("Blending: average of HAR, LightGBM and H-QLSTM", "Blending: HAR + LightGBM + H-QLSTM, average"),
      ("Blending: weighted average of the three", "Blending: HAR + LightGBM + H-QLSTM, weighted"),
      ("Stacking: HAR + LightGBM + H-QLSTM", "Stacking: HAR + LightGBM + H-QLSTM"),
      ("Blending: average, with the real twin (H-LSTM)", "Blending: HAR + LightGBM + H-LSTM, average")]),
    ("Three models, cascades (each model is trained on what the earlier ones leave)",
     [("HAR, then LightGBM (two stages)", "Cascade: HAR, then LightGBM"),
      ("HAR, then LightGBM, then H-QLSTM", "Cascade: HAR, then LightGBM, then H-QLSTM"),
      ("HAR, then LightGBM, then H-LSTM (real twin)", "Cascade: HAR, then LightGBM, then H-LSTM"),
      ("HAR, then H-QLSTM (two stages)", "Cascade: HAR, then H-QLSTM"),
      ("HAR, then H-QLSTM, then LightGBM", "Cascade: HAR, then H-QLSTM, then LightGBM")]),
    ("With the seasonal HAR in the place of HAR",
     [("Blending: average of seasonal HAR, LightGBM and H-QLSTM", "Blending: HAR-S x + LightGBM + H-QLSTM, average"),
      ("Blending: weighted average of the three", "Blending: HAR-S x + LightGBM + H-QLSTM, weighted"),
      ("Stacking: seasonal HAR + LightGBM + H-QLSTM", "Stacking: HAR-S x + LightGBM + H-QLSTM")]),
]


def fig():
    """Table for the advisor from tables/v3_triple.csv, drawn as an image (layout of paper_v3_runs.fig), with the
    MSE of every hybrid against that of the best two-model hybrid in the last column."""
    t = pd.read_csv(os.path.join(OUT, "v3_triple.csv")).set_index("model")
    last = {}
    for name, r in t.iterrows():
        if "vs_best" in t.columns and np.isfinite(r.get("vs_best", np.nan)):
            gap = round(100 * (r.vs_best - 1), 1)
            last[name] = ("0.0%" if gap == 0 else f"{gap:+.1f}%") + ("" if r.p_best >= 0.05 else " *")
    RU.draw(t, FIG_GROUPS, "Three-model hybrids of HAR, LightGBM and the H-QLSTM: five runs, mean ± standard deviation",
            "Next-bar log range, 15 cryptocurrencies, 4-hour bars, four test blocks (22 Apr 2025 - 14 Oct 2026, 48,600 forecasts). A run is one training seed (seeds 0-4); "
            "run k of a hybrid uses run k of each part.\nBlending weights and the stacking regression are fitted on the validation block of each fold. In a cascade every stage "
            "is trained on the residual of the earlier stages over the whole training block,\nwith out-of-fold forecasts of the earlier stages on the training rows. "
            "Last column: MSE of the average of the runs against the best two-model hybrid (positive = worse); * p < 0.05, cross-coin DM test. Bold: lowest mean.",
            FIG, highlight=(BEST,), extra=("MSE vs. best two-model", last))


if __name__ == "__main__":
    {"third": third, "fig": fig}.get((sys.argv[1:] or [""])[0], main)()
