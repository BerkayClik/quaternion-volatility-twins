"""
Five runs of every model, mean and standard deviation, MSE and QLIKE (asked for by the advisor, 2026-11-18:
whether the QLIKE scores are similar, and five runs of each model reported as mean +- standard deviation). Fixed
before the new runs are read:

  run        the network trained with seed k, k = 0..4 (one network, not an average of seeds). Seeds 0-4 of
             candidate A and its twin exist (ten were trained); seeds 3-4 of the benchmark networks, of the
             cascades and of the lagged LightGBM-S are new (configs/paper/v3/a_*_seeds_pool_4h.yaml;
             `lgbm` below). HAR and HAR-S x are OLS fits and have one value.
  hybrids    run k of a hybrid uses run k of each part: the plain average of H-QLSTM and LightGBM, the linear
             stack (fitted on the validation block of each fold), the plain averages with HAR and HAR-S x, and
             the cascades (stage 1, then run k of the network on its residual; stage 1 is the same in every run).
  MSE        over all coins and bars of the four test blocks, of the next-bar log range; also relative to HAR.
  QLIKE      as in the paper (paper_v2_round2._calibrated with the cap of train_caps): the variance forecast is
             k * exp(2 * forecast) with the level constant k from the validation block of the coin and fold, the
             realised squared range is capped at the largest value of the training block; mean over all coins
             and test bars; also relative to HAR. For the linear stack the validation forecasts are in sample.
  reported   mean and standard deviation (n - 1) over the runs, and the value of the average of the runs' forecasts
             (the paper reports averages of three seeds).

    .venv/bin/python -W ignore experiments/paper_v3_runs.py lgbm     seeds 3-4 of the lagged LightGBM-S (results/v3runs)
    .venv/bin/python -W ignore experiments/paper_v3_runs.py          tables/v3_runs.csv
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

OUT = "docs/paper/manuscript_v2/tables"
RUNS = (0, 1, 2, 3, 4)
LGBM_NEW = "results/v3runs/final_lgbmlags_{f}_s34"
V3 = "results/v3*/out/v3_a_{}_*"
A_RUNS = ["results/v3local/v3_gattn_range_pool_4h_*", V3.format("seeds_pool_4h")]
SEEDED = {   # name -> (key, run patterns)
    "LightGBM-S, lagged": (("PM", "lgbmsl-pool"), ["results/r8_local/final_lgbmlags_4h_r8", LGBM_NEW.format(f="4h")]),
    "H-QLSTM (candidate A)": (("PRMC", "hq_gattn-pool"), A_RUNS),
    "H-LSTM (real twin)": (("PRMC", "hl_gattn-pool"), A_RUNS),
    "Transformer": (("PRMC", "transformer-pool"), [V3.format("baselines_pool_4h"), V3.format("baselines_seeds_pool_4h")]),
    "iTransformer": (("PRMC", "itransformer-pool"), [V3.format("baselines_pool_4h"), V3.format("baselines_seeds_pool_4h")]),
    "PatchTST": (("PRMC", "patchtst-pool"), [V3.format("baselines_pool_4h"), V3.format("baselines_seeds_pool_4h")]),
    "CNN, 6 convolutions": (("PRMC", "cnn-L6-pool"), [V3.format("baselines_pool_4h"), V3.format("baselines_seeds_pool_4h")]),
    "QCNN, 6 convolutions": (("PRMC", "qcnn-L6-pool"), [V3.format("baselines_pool_4h"), V3.format("baselines_seeds_pool_4h")]),
    "DLinear": (("PRMC", "dlinear-pool"), [V3.format("baselines_pool_4h"), V3.format("baselines_seeds_pool_4h")]),
    "Cascade: LightGBM, then H-QLSTM": (("PRMC", "hq_gattn-pool"), [V3.format("cascade_pool_4h"), V3.format("cascade_seeds_pool_4h")]),
    "Cascade: LightGBM, then H-LSTM": (("PRMC", "hl_gattn-pool"), [V3.format("cascade_pool_4h"), V3.format("cascade_seeds_pool_4h")]),
    "Cascade: HAR, then H-QLSTM": (("PRMC", "hq_gattn-pool"), [V3.format("cascade_har_pool_4h"), V3.format("cascade_har_seeds_pool_4h")]),
    "Cascade: HAR-S x, then H-QLSTM": (("PRMC", "hq_gattn-pool"), [V3.format("cascade_harsx_pool_4h"), V3.format("cascade_harsx_seeds_pool_4h")]),
}
FIXED = {"HAR": (("P", "har"), ["results/final_fa/out/final_single_4h_*"]),
         "HAR-S x": (("P", "hars_x"), ["results/r8_local/final_harsx_4h_r8"])}
ORDER = ["HAR", "HAR-S x", "LightGBM-S, lagged", "DLinear", "QCNN, 6 convolutions", "CNN, 6 convolutions", "PatchTST", "iTransformer", "Transformer",
         "H-LSTM (real twin)", "H-QLSTM (candidate A)", "Blending: H-QLSTM + HAR", "Blending: H-QLSTM + HAR-S x", "Cascade: HAR, then H-QLSTM",
         "Cascade: HAR-S x, then H-QLSTM", "Cascade: LightGBM, then H-LSTM", "Cascade: LightGBM, then H-QLSTM", "Stacking: H-QLSTM + LightGBM",
         "Blending: H-LSTM + LightGBM", "Blending: H-QLSTM + LightGBM"]
IDX = ["dataset", "fold", "split", "time"]


def lgbm():
    """Seeds 3 and 4 of the lagged LightGBM-S, fitted exactly as seeds 0-2 (paper_round8_checks.lgbm_save), in a
    folder of their own so that the tables of paper v2 keep reading three seeds."""
    import paper_round8_checks as R8
    R8.LGBM_LAG_DIR = LGBM_NEW
    R8.lgbm_save(freqs=("4h",), seeds=(3, 4))


CELLS = 60      # coins x folds of a complete run; paper_v5_post sets 15 for its single fold


def forecasts(key: tuple, patterns: list[str]) -> dict:
    """seed -> Series(pred) indexed by (dataset, fold, split, time), validation and test forecasts; and y."""
    from src.paper.evaluate import _to_time
    parts = []
    for pattern in patterns:
        for directory in glob.glob(pattern):
            for path in glob.glob(os.path.join(directory, "preds", f"*__range__{key[0]}__{key[1]}__s*__f*.npz")):
                ds, _, _, _, seed, fold = os.path.basename(path)[:-4].split("__")
                z = np.load(path)
                for split in ("val", "test"):
                    parts.append(pd.DataFrame({"dataset": ds, "seed": int(seed[1:]), "fold": int(fold[1:]), "split": split,
                                               "time": _to_time(z[f"{split}_time"]), "y": z[f"{split}_y"].astype(float),
                                               "pred": z[f"{split}_pred"].astype(float)}))
    if not parts:
        return {}
    df = pd.concat(parts, ignore_index=True)
    out = {}
    for seed, g in df.groupby("seed"):
        if len(g[["dataset", "fold"]].drop_duplicates()) == CELLS:       # 15 coins x 4 folds: the run is complete
            out[int(seed)] = g.set_index(IDX)[["y", "pred"]].sort_index()
    return out


def losses(frame: pd.DataFrame, caps: pd.Series) -> tuple[float, float]:
    """MSE and QLIKE over all coins and test bars of one run (frame: y and pred indexed by IDX)."""
    import paper_v2_round2 as R
    ens = frame.reset_index().assign(groupset="g", model="m")
    test = ens[ens.split == "test"]
    assert len(test) == 48600, f"{len(test)} test coin-bars"
    ql = R._calibrated(ens, None, caps=caps).ql
    return float(((test.pred - test.y) ** 2).mean()), float(ql.mean())


def stack(a: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Linear stack of two runs: per fold an OLS of y on (1, a, g) on the validation block, applied to both blocks."""
    out = a.copy()
    x = pd.DataFrame({"y": a.y, "a": a.pred, "g": g.pred})
    for fold, f in x.groupby(level="fold"):
        va = f[f.index.get_level_values("split") == "val"]
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(va)), va.a, va.g]), va.y.to_numpy(), rcond=None)[0]
        out.loc[f.index, "pred"] = beta[0] + beta[1] * f.a + beta[2] * f.g
    return out


def main():
    import paper_v2_round2 as R
    caps = R.train_caps()
    runs = {name: forecasts(key, pats) for name, (key, pats) in SEEDED.items()}
    fixed = {name: forecasts(key, pats)[0] for name, (key, pats) in FIXED.items()}
    ref = fixed["HAR"]
    mix = lambda a, b: a.assign(pred=(a.pred + b.reindex(a.index).pred) / 2)
    A, T, G = runs["H-QLSTM (candidate A)"], runs["H-LSTM (real twin)"], runs["LightGBM-S, lagged"]
    both = [k for k in RUNS if k in A and k in G]
    runs["Blending: H-QLSTM + LightGBM"] = {k: mix(A[k], G[k]) for k in both}
    runs["Blending: H-LSTM + LightGBM"] = {k: mix(T[k], G[k]) for k in RUNS if k in T and k in G}
    runs["Stacking: H-QLSTM + LightGBM"] = {k: stack(A[k], G[k].reindex(A[k].index)) for k in both}
    runs["Blending: H-QLSTM + HAR"] = {k: mix(A[k], fixed["HAR"]) for k in RUNS if k in A}
    runs["Blending: H-QLSTM + HAR-S x"] = {k: mix(A[k], fixed["HAR-S x"]) for k in RUNS if k in A}
    mse_har, ql_har = losses(ref, caps)
    rows = []
    for name in ORDER:
        if name in fixed:
            mse, ql = losses(fixed[name], caps)
            rows.append(dict(model=name, runs=1, mse_mean=mse, mse_sd=np.nan, mse_har_mean=mse / mse_har, mse_har_sd=np.nan,
                             qlike_mean=ql, qlike_sd=np.nan, qlike_har_mean=ql / ql_har, qlike_har_sd=np.nan,
                             mse_har_avg=mse / mse_har, qlike_har_avg=ql / ql_har))
            continue
        have = [k for k in RUNS if k in runs[name]]
        if not have:
            print("not available:", name)
            continue
        for k in have:
            assert np.allclose(runs[name][k].y, ref.reindex(runs[name][k].index).y, atol=1e-4), f"{name}: targets differ from those of HAR"
        per = np.array([losses(runs[name][k], caps) for k in have])
        avg = runs[name][have[0]].assign(pred=np.mean([runs[name][k].reindex(runs[name][have[0]].index).pred for k in have], axis=0))
        mse_avg, ql_avg = losses(avg, caps)
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        rows.append(dict(model=name, runs=len(have), mse_mean=per[:, 0].mean(), mse_sd=sd(per[:, 0]),
                         mse_har_mean=(per[:, 0] / mse_har).mean(), mse_har_sd=sd(per[:, 0] / mse_har),
                         qlike_mean=per[:, 1].mean(), qlike_sd=sd(per[:, 1]),
                         qlike_har_mean=(per[:, 1] / ql_har).mean(), qlike_har_sd=sd(per[:, 1] / ql_har),
                         mse_har_avg=mse_avg / mse_har, qlike_har_avg=ql_avg / ql_har))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "v3_runs.csv"), index=False)
    pm = lambda m, s: f"{m:.4f}" + (f" ± {s:.4f}" if np.isfinite(s) else "")
    show = pd.DataFrame({"model": t.model, "runs": t.runs, "MSE": [pm(m, s) for m, s in zip(t.mse_mean, t.mse_sd)],
                         "MSE / HAR": [pm(m, s) for m, s in zip(t.mse_har_mean, t.mse_har_sd)],
                         "QLIKE": [pm(m, s) for m, s in zip(t.qlike_mean, t.qlike_sd)],
                         "QLIKE / HAR": [pm(m, s) for m, s in zip(t.qlike_har_mean, t.qlike_har_sd)],
                         "avg of runs: MSE / HAR": t.mse_har_avg.round(4), "QLIKE / HAR ": t.qlike_har_avg.round(4)})
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(show.to_string(index=False))


FIG = "docs/paper/review/advisor_2026-11-17/candidate_a/five_runs_table.png"
FIG_GROUPS = [   # (heading, [(label, row of tables/v3_runs.csv)])
    ("Benchmarks", [("HAR", "HAR"), ("Seasonal HAR (week cells)", "HAR-S x"), ("LightGBM-S, lagged features", "LightGBM-S, lagged")]),
    ("Networks on the same inputs", [("DLinear \u2020", "DLinear"), ("QCNN, 6 convolutions \u2020", "QCNN, 6 convolutions"),
                                     ("CNN, 6 convolutions \u2020", "CNN, 6 convolutions"), ("PatchTST", "PatchTST"),
                                     ("iTransformer", "iTransformer"), ("Transformer", "Transformer")]),
    ("Proposed network and its real twin", [("H-LSTM (real twin)", "H-LSTM (real twin)"), ("H-QLSTM (candidate A)", "H-QLSTM (candidate A)")]),
    ("H-QLSTM combined with HAR", [("Blending: average with HAR", "Blending: H-QLSTM + HAR"),
                                   ("Blending: average with seasonal HAR", "Blending: H-QLSTM + HAR-S x"),
                                   ("Cascade: HAR, then H-QLSTM", "Cascade: HAR, then H-QLSTM"),
                                   ("Cascade: seasonal HAR, then H-QLSTM", "Cascade: HAR-S x, then H-QLSTM")]),
    ("Combined with LightGBM", [("Cascade: LightGBM, then H-LSTM", "Cascade: LightGBM, then H-LSTM"),
                                ("Cascade: LightGBM, then H-QLSTM", "Cascade: LightGBM, then H-QLSTM"),
                                ("Stacking: H-QLSTM + LightGBM", "Stacking: H-QLSTM + LightGBM"),
                                ("Blending: average of H-LSTM and LightGBM", "Blending: H-LSTM + LightGBM"),
                                ("Blending: average of H-QLSTM and LightGBM", "Blending: H-QLSTM + LightGBM")]),
]


def _pm(mean: float, sd: float, digits: int) -> str:
    """mean ± sd with `digits` decimals; fewer when the value is large (a loss that a few bars have blown up)."""
    d = 0 if mean >= 10 else 2 if mean >= 1.5 else digits
    return f"{mean:.{d}f}" + (f" ± {sd:.{d}f}" if np.isfinite(sd) else "")


def draw(t: pd.DataFrame, groups: list, title: str, note: str, path: str, highlight: tuple = (), extra: tuple | None = None):
    """A table of results as an image: one row per model, mean ± standard deviation over the runs of MSE and QLIKE,
    in the units of the loss and relative to HAR. t: a table like tables/v3_runs.csv indexed by model; groups:
    (heading, [(label, model)]); highlight: models on a tinted band; extra: (heading, {model: text}), a last column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})
    INK, MUTED, RULE, BAND = "#1a1a1a", "#6b6b6b", "#c9c9c9", "#f3f6f9"
    if extra is None:
        cols, width = [("Runs", 0.405, "center"), ("MSE", 0.560, "right"), ("MSE / HAR", 0.700, "right"), ("QLIKE", 0.850, "right"), ("QLIKE / HAR", 0.990, "right")], 10.2
    else:
        cols, width = [("Runs", 0.375, "center"), ("MSE", 0.505, "right"), ("MSE / HAR", 0.625, "right"), ("QLIKE", 0.735, "right"),
                       ("QLIKE / HAR", 0.845, "right"), (extra[0], 0.995, "right")], 11.8
    n_rows = sum(len(rows) for _, rows in groups) + len(groups)
    h = 0.285
    fig_, ax = plt.subplots(figsize=(width, h * (n_rows + 1) + 1.55))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n_rows + 1.2)
    shown = [name for _, rows in groups for _, name in rows if name in t.index]
    best = {c: t.loc[shown, c].idxmin() for c in ("mse_mean", "qlike_mean")}
    y = n_rows + 0.55
    ax.text(0.0, y, "Model", fontsize=9, color=MUTED, va="center")
    for label, x, ha in cols:
        ax.text(x, y, label, fontsize=9, color=MUTED, va="center", ha=ha)
    ax.plot([0, 1], [y - 0.5, y - 0.5], color=INK, lw=0.8)
    y -= 0.05
    for heading, rows in groups:
        y -= 1
        ax.text(0.0, y, heading, fontsize=9, fontweight="bold", color=INK, va="center")
        ax.plot([0, 1], [y - 0.45, y - 0.45], color=RULE, lw=0.6)
        for label, name in rows:
            y -= 1
            if name not in t.index:
                ax.text(0.015, y, label, fontsize=9, color=MUTED, va="center")
                ax.text(cols[1][1], y, "runs not finished", fontsize=8.5, color=MUTED, va="center", ha="right")
                continue
            r = t.loc[name]
            if name in highlight:
                ax.add_patch(plt.Rectangle((0, y - 0.47), 1, 0.94, color=BAND, zorder=0, lw=0))
            ax.text(0.015, y, label, fontsize=9, color=INK, va="center")
            cells = [str(int(r.runs)), _pm(r.mse_mean, r.mse_sd, 4), _pm(r.mse_har_mean, r.mse_har_sd, 4),
                     _pm(r.qlike_mean, r.qlike_sd, 3), _pm(r.qlike_har_mean, r.qlike_har_sd, 3)]
            bold = [False, best["mse_mean"] == name, best["mse_mean"] == name, best["qlike_mean"] == name, best["qlike_mean"] == name]
            if extra is not None:
                cells.append(extra[1].get(name, "")); bold.append(False)
            for (_, x, ha), text, b in zip(cols, cells, bold):
                if extra is not None and x == cols[-1][1]:       # the mark of significance stands right of the aligned numbers
                    ax.text(x - 0.004, y, " *" if text.endswith(" *") else "", fontsize=9, color=INK, va="center", ha="left")
                    x, text = x - 0.004, text.removesuffix(" *")
                ax.text(x, y, text, fontsize=9, color=INK, va="center", ha=ha, fontweight="bold" if b else "normal")
    ax.plot([0, 1], [y - 0.55, y - 0.55], color=INK, lw=0.8)
    fig_.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig_.text(0.012, 0.012, note, fontsize=7.2, color=MUTED, ha="left", linespacing=1.4)
    fig_.subplots_adjust(left=0.012, right=0.988, top=0.945, bottom=0.078)
    fig_.savefig(path, dpi=220, facecolor="white")
    plt.close(fig_)
    print("written", path)


def fig():
    """Table for the advisor from tables/v3_runs.csv, drawn as an image."""
    draw(pd.read_csv(os.path.join(OUT, "v3_runs.csv")).set_index("model"), FIG_GROUPS,
         "Five runs per model: mean ± standard deviation of MSE and QLIKE",
         "Next-bar log range, 15 cryptocurrencies, 4-hour bars, four test blocks (22 Apr 2025 - 14 Oct 2026, 48,600 forecasts). A run is one training seed (seeds 0-4); "
         "HAR and seasonal HAR are OLS fits with one value.\nIn a combination, run k uses run k of each part. QLIKE of the variance forecast, level constant from the validation "
         "block, realised squared range capped at the training maximum. Bold: lowest mean.\n\u2020 The mean QLIKE of these models is carried by a few bars with far too low "
         "forecasts (in one QCNN run ten of 48,600 terms hold 98% of the sum); their median QLIKE is close to that of the other networks.",
         FIG, highlight=("Blending: H-QLSTM + LightGBM", "H-QLSTM (candidate A)"))


if __name__ == "__main__":
    {"lgbm": lgbm, "fig": fig}.get((sys.argv[1:] or [""])[0], main)()
