"""
Further benchmarks and tuned benchmarks for manuscript v4 (review of 2026-11-23, items M2 and M5).

Plan (2026-11-23, fixed before any of these models is fitted; committed before the runs). Known before: every table
of manuscript v4 and of the second round (tables/v4_r2_*.csv); for the two versions of the network the learning
rates 3e-3 and 3e-4 with three seeds (tables/v3_grid.csv: no significant change for the real version). Data, folds,
inputs, parameter budget and seeds 0 to 4 are those of the paper; "the hybrid" is H-LSTM + LightGBM.

  garch      GARCH(1,1) and GJR-GARCH(1,1,1) with Student-t errors on the four-hour returns of each coin, fitted on
             the training block; the forecast of the log range is a regression of the log range on the log of the
             one-step conditional standard deviation, fitted on the training block (src/paper/baselines.py, "garch"
             and "gjr"). results/v4bench/garch.
  hgb        a second implementation of gradient-boosted trees on the 35 regressors of the lagged LightGBM-S:
             HistGradientBoostingRegressor of scikit-learn (learning rate 0.03, 31 leaves, at least 100 observations
             per leaf, L2 penalty 1, 80% of the features per split, at most 3,000 trees, the number of trees with
             the lowest MSE on the validation block), pooled, five seeds. results/v4bench/hgb.
             Averages: LightGBM + HGB (one view, two libraries) and H-LSTM + HGB (two views, the other library).
  lgbm_grid  the lagged LightGBM-S with 31 / 15 / 63 leaves, learning rate 0.03 / 0.1 and at least 100 / 50 / 200
             observations per leaf (18 settings, the first of each list is the setting of the paper), five seeds.
             results/v4bench/lgbm_grid/<setting>.
  networks   (Colab, configs/paper/v4/tune_*.yaml) learning rates 3e-3 and 3e-4 next to the 1e-3 of the paper for the
             six benchmark networks and for the two versions of the network of the hybrid; PatchTST and
             iTransformer in addition with per-window instance normalisation (+in), their native setting, at the
             three learning rates. Five seeds each.
  tuned      For a model family the tuned forecast of run k is, in every fold, the forecast of the setting with the
             lowest MSE of run k on the validation block of that fold (settings: three learning rates; six for
             PatchTST and iTransformer; 18 for LightGBM). The tuned hybrid is the plain average of the tuned H-LSTM
             and the tuned LightGBM, run k with run k.
  tables     tables/v4_bench.csv: GARCH, GJR, HGB and the two averages with HGB, MSE and QLIKE, each against the
             hybrid with the two tests of the paper. tables/v4_tuned.csv: every family with its default and its
             tuned MSE (mean and standard deviation over five runs), the settings chosen, the tuned hybrid against
             every tuned single model and the default hybrid against every tuned single model, with the two tests.
  rule       The main result holds under tuning if the tuned hybrid has a lower MSE than every tuned single model by
             the rule of the paper (both p < 0.05). Reported whatever it is.

    .venv/bin/python -W ignore experiments/paper_v4_bench.py [garch|hgb|lgbm_grid|eval]
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

FREQ, SEEDS, ROOT = "4h", (0, 1, 2, 3, 4), "results/v4bench"
GRID = [dict(num_leaves=nl, learning_rate=lr, min_child_samples=mc) for nl, lr, mc in itertools.product((31, 15, 63), (0.03, 0.1), (100, 50, 200))]
tag = lambda g: f"leaves{g['num_leaves']}_lr{g['learning_rate']}_min{g['min_child_samples']}"


def _save(out: str, name: str, ds: str, gs: str, seed: int, fold: int, frame: pd.DataFrame, res: dict):
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    if not os.path.exists(os.path.join(out, "config.json")):
        json.dump({"config": {"name": os.path.basename(out), "note": "paper_v4_bench"}}, open(os.path.join(out, "config.json"), "w"), indent=1)
    idx = frame.index if frame.index.tz is None else frame.index.tz_convert(None)
    t = idx.as_unit("ns").asi8
    y = frame["y_range"].to_numpy(float)
    np.savez(os.path.join(out, "preds", f"{ds}__range__{gs}__{name}__s{seed}__f{fold}.npz"),
             test_time=t[res["test_rows"]], test_y=y[res["test_rows"]].astype("float32"), test_pred=np.asarray(res["test_pred"], "float32"),
             val_time=t[res["val_rows"]], val_y=y[res["val_rows"]].astype("float32"), val_pred=np.asarray(res["val_pred"], "float32"),
             meta=np.array(json.dumps(dict(freq=FREQ, seed=seed, fold=fold, model=name))))


def garch():
    from paper_round7_checks import _frames
    from src.paper.baselines import predict_baseline
    fr, win = _frames(FREQ)
    for ds, (d, folds) in fr.items():
        for k, fold in enumerate(folds):
            for name in ("garch", "gjr"):
                _save(f"{ROOT}/garch", name, ds, "P", 0, k, d.frame, predict_baseline(name, d.frame, "range", fold, win, FREQ))
        print(ds, flush=True)


def _pooled(out: str, name: str, seeds=SEEDS):
    """The lagged design of the paper's LightGBM-S (paper_round8_checks.lgbm_save) with the fitting routine that
    src.paper.baselines._lgbm holds at the time of the call."""
    from paper_round7_checks import _cfg, _frames
    from src.paper.baselines import HAR_LAGS, predict_baseline_pooled
    fr, win = _frames(FREQ)
    dss = list(fr)
    Xs = [fr[ds][0].X(_cfg(f"lgbm_{FREQ}")["groupsets"]["PM"]) for ds in dss]
    Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS[FREQ][1:]]) for X in Xs]
    for seed in seeds:
        for k in range(len(fr[dss[0]][1])):
            if all(os.path.exists(os.path.join(out, "preds", f"{ds}__range__PM__{name}__s{seed}__f{k}.npz")) for ds in dss):
                continue
            res = predict_baseline_pooled("lgbms", [fr[ds][0].frame for ds in dss], "range", [fr[ds][1][k] for ds in dss], win, FREQ, Xl, seed=seed)
            for ds, r in zip(dss, res):
                _save(out, name, ds, "PM", seed, k, fr[ds][0].frame, r)
        print(out, "seed", seed, flush=True)


def hgb():
    import lightgbm  # noqa: F401  (its OpenMP runtime first, as in the runner)
    from sklearn.ensemble import HistGradientBoostingRegressor
    import src.paper.baselines as B

    class Staged:
        """The fitted model, truncated at the number of trees with the lowest MSE on the validation block."""

        def __init__(self, m, n):
            self.m, self.n = m, n

        def predict(self, X):
            for i, p in enumerate(self.m.staged_predict(X), start=1):
                if i == self.n:
                    return p

    def fit(Xtr, ytr, Xva, yva, seed=0):
        m = HistGradientBoostingRegressor(learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, max_features=0.8,
                                          max_iter=3000, early_stopping=False, random_state=seed).fit(Xtr, ytr)
        mse = [float(np.mean((p - yva) ** 2)) for p in m.staged_predict(Xva)]
        return Staged(m, int(np.argmin(mse)) + 1)

    B._lgbm = fit
    _pooled(f"{ROOT}/hgb", "hgbsl-pool")


def lgbm_grid():
    import lightgbm  # noqa: F401
    import src.paper.baselines as B
    default = dict(B.LGBM_PARAMS)
    for g in GRID:
        B.LGBM_PARAMS = {**default, **g}
        _pooled(f"{ROOT}/lgbm_grid/{tag(g)}", "lgbmsl-pool")


def _families() -> dict:
    """family -> {setting: (key, run patterns)}; the first setting of every family is that of the paper."""
    import paper_v3_runs as RU
    V3, T = RU.V3, "results/v4tune*/out/v4_{}_*"
    fam = {}
    for label, name in (("DLinear", "DLinear"), ("CNN", "CNN, 6 convolutions"), ("QCNN", "QCNN, 6 convolutions"), ("PatchTST", "PatchTST"),
                        ("iTransformer", "iTransformer"), ("Transformer", "Transformer")):
        key, pats = RU.SEEDED[name]
        fam[label] = {"lr 1e-3": (key, pats), "lr 3e-3": (key, [T.format("tune_bench_lr3e3_pool_4h")]), "lr 3e-4": (key, [T.format("tune_bench_lr3e4_pool_4h")])}
        if label in ("PatchTST", "iTransformer"):
            kin = (key[0], key[1].replace("-pool", "+in-pool"))
            fam[label].update({f"+in, lr {lr}": (kin, [T.format(f"tune_in_lr{t}_pool_4h")]) for lr, t in (("1e-3", "1e3"), ("3e-3", "3e3"), ("3e-4", "3e4"))})
    for label, name in (("H-QLSTM", "H-QLSTM (candidate A)"), ("H-LSTM", "H-LSTM (real twin)")):
        key, pats = RU.SEEDED[name]
        fam[label] = {"lr 1e-3": (key, pats), "lr 3e-3": (key, [V3.format("lr3e3_pool_4h"), T.format("tune_net_lr3e3_pool_4h")]),
                      "lr 3e-4": (key, [V3.format("lr3e4_pool_4h"), T.format("tune_net_lr3e4_pool_4h")])}
    fam["LightGBM"] = {tag(g): (("PM", "lgbmsl-pool"), [f"{ROOT}/lgbm_grid/{tag(g)}"]) for g in GRID}
    return fam


def evaluate():
    import paper_v2_round2 as R
    import paper_v3_paper as P3
    import paper_v3_runs as RU
    from paper_v3_paper import LGBM, compare
    from paper_v4_paper import _losses
    CSV = RU.OUT
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    get = RU.forecasts
    M0, _ = P3.build()
    HYB, NET = P3.TWIN_HYBRID, P3.TWIN
    ref = M0[NET][0].index
    fold = ref.get_level_values("fold").to_numpy()
    is_val = ref.get_level_values("split").to_numpy() == "val"
    y = M0[NET][0].y.to_numpy()
    frame = lambda pred: pd.DataFrame({"y": y, "pred": np.asarray(pred)}, index=ref)
    mix = lambda a, b: frame((a.pred.to_numpy() + b.pred.to_numpy()) / 2)

    # further benchmarks
    B = {"GARCH(1,1)": {0: get(("P", "garch"), [f"{ROOT}/garch"])[0].reindex(ref)}, "GJR-GARCH": {0: get(("P", "gjr"), [f"{ROOT}/garch"])[0].reindex(ref)},
         "HGB": {k: f.reindex(ref) for k, f in get(("PM", "hgbsl-pool"), [f"{ROOT}/hgb"]).items()}}
    B["LightGBM + HGB"] = {k: mix(M0[LGBM][k].reindex(ref), B["HGB"][k]) for k in SEEDS}
    B["H-LSTM + HGB"] = {k: mix(M0[NET][k], B["HGB"][k]) for k in SEEDS}
    per_run, per_bar = _losses({HYB: M0[HYB], LGBM: M0[LGBM], **B}, R.train_caps())
    har = pd.read_csv(os.path.join(CSV, "v3_paper_models.csv")).set_index("model").loc["HAR", "mse_mean"]
    rows = [dict(model=n, runs=len(per_run[n]), mse_mean=per_run[n].mse.mean(), mse_sd=sd(per_run[n].mse), mse_har=per_run[n].mse.mean() / har,
                 qlike_mean=per_run[n].qlike.mean(), qlike_sd=sd(per_run[n].qlike), **{"hyb_" + c: v for c, v in compare(per_run, per_bar, HYB, n).items()},
                 **{"lgbm_" + c: v for c, v in compare(per_run, per_bar, n, LGBM).items() if c.startswith("mse_")}) for n in B]
    bench = pd.DataFrame(rows)
    bench.to_csv(os.path.join(CSV, "v4_bench.csv"), index=False)

    # tuned families
    val_mse = lambda pred, f: float(np.mean((pred[is_val & (fold == f)] - y[is_val & (fold == f)]) ** 2))
    tuned, default, choices, per_setting = {}, {}, [], []
    for label, settings in _families().items():
        runs = {}
        for s_, (key, pats) in settings.items():
            r = get(key, pats)
            if not set(SEEDS) <= set(r):                 # the paper's two networks have ten seeds; seeds 0-4 are used
                print(f"incomplete: {label}, {s_}: runs {sorted(r)}")
                continue
            runs[s_] = {k: r[k].reindex(ref).pred.to_numpy() for k in SEEDS}
            assert all(np.isfinite(v).all() for v in runs[s_].values()), (label, s_)
        for s_ in runs:                                    # test MSE of every setting, for the description of the grid
            te = ~is_val
            per_setting.append(dict(family=label, setting=s_, test_mse=float(np.mean([np.mean((runs[s_][k][te] - y[te]) ** 2) for k in SEEDS])),
                                    val_mse=float(np.mean([np.mean((runs[s_][k][is_val] - y[is_val]) ** 2) for k in SEEDS]))))
        first = next(iter(settings))
        default[label] = {k: frame(runs[first][k]) for k in SEEDS}
        tuned[label] = {}
        for k in SEEDS:
            pred = np.full(len(ref), np.nan)
            for f in np.unique(fold):
                best = min(runs, key=lambda s_: val_mse(runs[s_][k], f))
                pred[fold == f] = runs[best][k][fold == f]
                choices.append(dict(family=label, run=k, fold=int(f), setting=best))
            tuned[label][k] = frame(pred)
        print(label, "settings:", len(runs), flush=True)
    fixed = {n: {0: M0[n][0].reindex(ref)} for n in ("HAR", "HAR-S x")}
    allm = {**{f"{n} (default)": v for n, v in default.items()}, **{f"{n} (tuned)": v for n, v in tuned.items()}, **fixed,
            "Hybrid (default)": M0[HYB], "Hybrid (tuned)": {k: mix(tuned["H-LSTM"][k], tuned["LightGBM"][k]) for k in SEEDS},
            "Hybrid, quaternion (tuned)": {k: mix(tuned["H-QLSTM"][k], tuned["LightGBM"][k]) for k in SEEDS}}
    per_run, per_bar = _losses(allm, R.train_caps())
    rows = []
    for label in list(tuned) + ["HAR", "HAR-S x"]:
        t_name, d_name = (f"{label} (tuned)", f"{label} (default)") if label in tuned else (label, label)
        pt, pdf = per_run[t_name], per_run[d_name]
        r = dict(family=label, default_mse=pdf.mse.mean(), default_sd=sd(pdf.mse), tuned_mse=pt.mse.mean(), tuned_sd=sd(pt.mse), tuned_qlike=pt.qlike.mean())
        if label in tuned:
            r.update({"gain_" + c: v for c, v in compare(per_run, per_bar, t_name, d_name).items() if c.startswith("mse_")})
        r.update({"tunedhyb_" + c: v for c, v in compare(per_run, per_bar, "Hybrid (tuned)", t_name).items()})
        r.update({"defaulthyb_" + c: v for c, v in compare(per_run, per_bar, "Hybrid (default)", t_name).items() if c.startswith("mse_")})
        rows.append(r)
    for name in ("Hybrid (default)", "Hybrid (tuned)", "Hybrid, quaternion (tuned)"):
        p_ = per_run[name]
        r = dict(family=name, tuned_mse=p_.mse.mean(), tuned_sd=sd(p_.mse), tuned_qlike=p_.qlike.mean())
        if name != "Hybrid (tuned)":
            r.update({"tunedhyb_" + c: v for c, v in compare(per_run, per_bar, "Hybrid (tuned)", name).items()})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v4_tuned.csv"), index=False)
    pd.DataFrame(per_setting).to_csv(os.path.join(CSV, "v4_tuned_settings.csv"), index=False)
    c = pd.DataFrame(choices)
    c.groupby(["family", "setting"]).size().rename("fits").reset_index().to_csv(os.path.join(CSV, "v4_tuned_choices.csv"), index=False)
    pd.set_option("display.width", 280, "display.max_columns", 40, "display.max_rows", 200)
    print(bench[["model", "runs", "mse_mean", "mse_sd", "mse_har", "qlike_mean", "hyb_mse_diff_pct", "hyb_mse_p_runs", "hyb_mse_p_time", "hyb_mse_coins_better",
                 "hyb_mse_verdict", "hyb_qlike_diff_pct", "hyb_qlike_verdict", "lgbm_mse_diff_pct", "lgbm_mse_verdict"]].round(4).to_string(index=False))
    print(t[["family", "default_mse", "tuned_mse", "tuned_sd", "gain_mse_diff_pct", "gain_mse_verdict", "tunedhyb_mse_diff_pct", "tunedhyb_mse_p_runs",
             "tunedhyb_mse_p_time", "tunedhyb_mse_coins_better", "tunedhyb_mse_verdict", "defaulthyb_mse_diff_pct", "defaulthyb_mse_verdict", "tunedhyb_qlike_diff_pct",
             "tunedhyb_qlike_verdict"]].round(4).to_string(index=False))
    print(c.groupby(["family", "setting"]).size().rename("fits").reset_index().to_string(index=False))
    singles = t[t.family.isin(list(tuned) + ["HAR", "HAR-S x"])]
    print("RULE (tuned hybrid better than every tuned single model):", "HOLDS" if (singles.tunedhyb_mse_verdict == "better").all()
          else f"DOES NOT HOLD: {singles[singles.tunedhyb_mse_verdict != 'better'].family.tolist()}")


if __name__ == "__main__":
    {"garch": garch, "hgb": hgb, "lgbm_grid": lgbm_grid, "eval": evaluate}[sys.argv[1]]()
