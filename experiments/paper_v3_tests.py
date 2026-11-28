"""
Our method against the other methods, with statistical tests (asked for by the advisor, 2026-11-20: the
improvement may look minor, but with standard deviations this small a statistical test may find it significant;
compare the result of our method with the other methods statistically). No new run: the five runs of every model of
tables/v3_runs.csv (experiments/paper_v3_runs.py). Fixed before any of these tests was computed. Known before: the
means and standard deviations of that table, and DM tests on averages of the runs' forecasts (v3_hybrid.csv,
v3_triple.csv).

  ours        (a) the hybrid: the plain average of the H-QLSTM and the lagged LightGBM-S, run k with run k;
              (b) the H-QLSTM alone (candidate A).
  others      HAR, the seasonal HAR, the lagged LightGBM-S, DLinear, QCNN, CNN, PatchTST, iTransformer, Transformer,
              the real twin (H-LSTM), the hybrid with the real twin in the place of the H-QLSTM, and the other one
              of our two methods.
  losses      MSE and QLIKE of a run as in paper_v3_runs (48,600 forecasts of the four test blocks).
  two tests   they answer two questions, and a difference has to pass both:
    over runs   is the difference larger than what the training seed moves? Welch's t-test (unequal variances,
                two-sided) on the five losses of our runs against the five of the other model; against HAR and the
                seasonal HAR, which have one value, the one-sample t-test. Beside it the exact Mann-Whitney U test
                (five against five: the smallest possible two-sided p is 0.008). A hybrid and its parts share
                seeds; they are treated as independent samples all the same, which is the cautious side.
    over time   is the difference larger than what the test period moves? Diebold-Mariano test on the loss
                difference of every bar, averaged over the five runs and over the coins (Newey-West variance, as
                in src/paper/evaluate.dm_cross). The mean of that series is the difference of the two means of the
                table, so that both tests are about the same number; 90% interval from the same variance.
  adjusted    Holm over the comparisons of one of our two methods, per test and loss (in the CSV file).
  rule        Our method counts as better than another one in a loss only if its mean is lower and both the t-test
              and the DM test give p < 0.05; as worse if its mean is higher and both give p < 0.05. Every row is
              reported, with "runs only" or "time only" where one test alone is below 0.05.

    .venv/bin/python -W ignore experiments/paper_v3_tests.py        tables/v3_tests.csv
    .venv/bin/python -W ignore experiments/paper_v3_tests.py fig    review/advisor_2026-11-17/candidate_a/significance_table.png
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_runs as RU  # noqa: E402

OUT, RUNS = RU.OUT, RU.RUNS
HYBRID, NET, TWIN, LGBM, TWIN_HYBRID = "H-QLSTM + LightGBM", "H-QLSTM", "H-LSTM (real twin)", "LightGBM-S, lagged", "H-LSTM + LightGBM"
OTHERS = ["HAR", "HAR-S x", LGBM, "DLinear", "QCNN, 6 convolutions", "CNN, 6 convolutions", "PatchTST", "iTransformer", "Transformer",
          TWIN, TWIN_HYBRID]
OURS = [(HYBRID, OTHERS + [NET]), (NET, OTHERS + [HYBRID])]


N_TEST = 48600      # test forecasts of a data set (15 coins, four blocks of 135 days); paper_v5_post sets 3,990 for its single block


def bar_losses(frame: pd.DataFrame, caps: pd.Series) -> pd.DataFrame:
    """Squared error and QLIKE of every test forecast of one run, indexed by (time, dataset)."""
    import paper_v2_round2 as R
    ens = frame.reset_index().assign(groupset="g", model="m")
    test = R._calibrated(ens, None, caps=caps)
    assert len(test) == N_TEST, f"{len(test)} test coin-bars"
    return pd.DataFrame({"mse": (test.pred - test.y) ** 2, "qlike": test.ql}).set_index([test.time, test.dataset]).sort_index()


def verdict(diff: float, p_runs: float, p_time: float) -> str:
    runs, time_ = p_runs < 0.05, p_time < 0.05
    if runs and time_:
        return "better" if diff < 0 else "worse"
    return "runs only" if runs else "time only" if time_ else "no difference"


def main():
    import paper_v2_round2 as R
    from scipy import stats
    from src.paper.evaluate import diebold_mariano, holm
    caps = R.train_caps()
    runs = {name: RU.forecasts(key, pats) for name, (key, pats) in RU.SEEDED.items()}
    A, T, G = runs["H-QLSTM (candidate A)"], runs["H-LSTM (real twin)"], runs[LGBM]
    mix = lambda a, b: a.assign(pred=(a.pred + b.reindex(a.index).pred) / 2)
    models = {name: {k: runs[name][k] for k in RUNS} for name in OTHERS if name in runs}
    models.update({name: {0: RU.forecasts(*RU.FIXED[name])[0]} for name in ("HAR", "HAR-S x")})
    models[NET], models[TWIN] = {k: A[k] for k in RUNS}, {k: T[k] for k in RUNS}
    models[HYBRID], models[TWIN_HYBRID] = {k: mix(A[k], G[k]) for k in RUNS}, {k: mix(T[k], G[k]) for k in RUNS}
    per_run, per_bar = {}, {}                  # model -> (runs x 2) losses of the runs; model -> per-bar losses, mean over the runs
    for name, by_run in models.items():
        bars = [bar_losses(f, caps) for f in by_run.values()]
        per_run[name] = pd.DataFrame([b.mean() for b in bars])
        per_bar[name] = sum(bars) / len(bars)
        print(f"{name:26s} runs {len(bars)}  MSE {per_run[name].mse.mean():.4f}  QLIKE {per_run[name].qlike.mean():.4f}", flush=True)
    rows = []
    for ours, others in OURS:
        for other in others:
            r = dict(ours=ours, other=other, runs_ours=len(per_run[ours]), runs_other=len(per_run[other]))
            for loss in ("mse", "qlike"):
                x, y = per_run[ours][loss].to_numpy(), per_run[other][loss].to_numpy()
                if len(y) > 1:
                    t_stat, p_t = stats.ttest_ind(x, y, equal_var=False)
                    p_u = stats.mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue
                else:
                    (t_stat, p_t), p_u = stats.ttest_1samp(x, y[0]), np.nan
                d = (per_bar[ours][loss] - per_bar[other][loss]).unstack("dataset")
                series = d.mean(axis=1).sort_index().to_numpy()
                dm, p_dm = diebold_mariano(series, np.zeros(len(series)))
                se = abs(series.mean() / dm)
                coin = d.mean(axis=0)
                assert np.isclose(series.mean(), x.mean() - y.mean(), rtol=1e-6, atol=1e-9), (ours, other, loss)
                r.update({f"{loss}_ours": x.mean(), f"{loss}_ours_sd": np.std(x, ddof=1), f"{loss}_other": y.mean(),
                          f"{loss}_other_sd": np.std(y, ddof=1) if len(y) > 1 else np.nan,
                          f"{loss}_diff_pct": 100 * (x.mean() / y.mean() - 1),
                          f"{loss}_diff_lo": 100 * (series.mean() - 1.645 * se) / y.mean(), f"{loss}_diff_hi": 100 * (series.mean() + 1.645 * se) / y.mean(),
                          f"{loss}_coins_better": int((coin < 0).sum()), f"{loss}_t": t_stat, f"{loss}_p_runs": p_t, f"{loss}_p_runs_u": p_u,
                          f"{loss}_dm": dm, f"{loss}_p_time": p_dm, f"{loss}_verdict": verdict(x.mean() - y.mean(), p_t, p_dm)})
            rows.append(r)
    t = pd.DataFrame(rows)
    for ours, g in t.groupby("ours"):
        for col in ("mse_p_runs", "mse_p_time", "qlike_p_runs", "qlike_p_time"):
            t.loc[g.index, col + "_holm"] = holm(g[col].to_numpy())
    t.to_csv(os.path.join(OUT, "v3_tests.csv"), index=False)
    pd.set_option("display.width", 280, "display.max_columns", 30)
    show = t[["ours", "other", "mse_other", "mse_diff_pct", "mse_p_runs", "mse_p_runs_u", "mse_p_time", "mse_coins_better", "mse_verdict",
              "qlike_other", "qlike_diff_pct", "qlike_p_runs", "qlike_p_time", "qlike_coins_better", "qlike_verdict"]]
    print(show.round(4).to_string(index=False))
    print(t[["ours", "other", "mse_p_runs_holm", "mse_p_time_holm", "qlike_p_runs_holm", "qlike_p_time_holm"]].round(4).to_string(index=False))


FIG = "docs/paper/review/advisor_2026-11-17/candidate_a/significance_table.png"
LABEL = {"HAR": "HAR", "HAR-S x": "Seasonal HAR (week cells)", LGBM: "LightGBM-S, lagged features", "DLinear": "DLinear †",
         "QCNN, 6 convolutions": "QCNN, 6 convolutions †", "CNN, 6 convolutions": "CNN, 6 convolutions †", "PatchTST": "PatchTST",
         "iTransformer": "iTransformer", "Transformer": "Transformer", TWIN: "H-LSTM (real twin of the H-QLSTM)",
         TWIN_HYBRID: "H-LSTM + LightGBM (hybrid with the real twin)", NET: "H-QLSTM alone", HYBRID: "H-QLSTM + LightGBM (the hybrid)"}
WORD = {"better": "ours better", "worse": "ours worse", "no difference": "no difference", "runs only": "runs only", "time only": "time only"}


def fig():
    """Table for the advisor from tables/v3_tests.csv, drawn as an image: one block per method of ours, one row per
    other method; for MSE and QLIKE the loss of the other method, the difference of ours in percent, the p-values of
    the test over runs and of the test over time, and the verdict of the rule."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})
    INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#c9c9c9"
    t = pd.read_csv(os.path.join(OUT, "v3_tests.csv"))
    p_text = lambda p: "" if not np.isfinite(p) else "<0.001" if p < 0.001 else f"{p:.3f}"
    left, half = 0.245, (1 - 0.245) / 2
    sub = [("loss, mean ± sd", 0.36, "right"), ("ours vs. it", 0.52, "right"), ("p, runs", 0.66, "right"), ("p, time", 0.79, "right"), ("result", 0.995, "right")]
    n_rows = len(t) + 2 * len(OURS)
    fig_, ax = plt.subplots(figsize=(13.6, 0.285 * (n_rows + 1) + 1.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(2.2, n_rows + 3.2)
    y = n_rows + 2.6
    ax.text(0.0, y - 0.5, "Compared with", fontsize=9, color=MUTED, va="center")
    for j, loss in enumerate(("MSE", "QLIKE")):
        x0 = left + half * j
        ax.text(x0 + half * 0.52, y, loss, fontsize=9, fontweight="bold", color=INK, va="center", ha="center")
        ax.plot([x0 + 0.012, x0 + half - 0.004], [y - 0.5, y - 0.5], color=RULE, lw=0.6)
        for label, x, ha in sub:
            ax.text(x0 + half * x, y - 1, label, fontsize=8.5, color=MUTED, va="center", ha=ha)
    y -= 1
    ax.plot([0, 1], [y - 0.5, y - 0.5], color=INK, lw=0.8)
    y -= 0.05
    for ours, _ in OURS:
        g = t[t.ours == ours]
        first = g.iloc[0]
        y -= 1.25
        ax.text(0.0, y, f"Ours: {LABEL[ours]}", fontsize=9, fontweight="bold", color=INK, va="center")
        for j, loss in enumerate(("mse", "qlike")):
            ax.text(left + half * (j + sub[0][1]), y, RU._pm(first[f"{loss}_ours"], first[f"{loss}_ours_sd"], 4 if loss == "mse" else 3),
                    fontsize=9, fontweight="bold", color=INK, va="center", ha="right")
        ax.plot([0, 1], [y - 0.45, y - 0.45], color=RULE, lw=0.6)
        for _, r in g.iterrows():
            y -= 1
            ax.text(0.014, y, LABEL[r.other], fontsize=9, color=INK, va="center")
            for j, loss in enumerate(("mse", "qlike")):
                x0, word = left + half * j, r[f"{loss}_verdict"]
                weight = "bold" if word in ("better", "worse") else "normal"
                gap = r[f"{loss}_diff_pct"]
                cells = [RU._pm(r[f"{loss}_other"], r[f"{loss}_other_sd"], 4 if loss == "mse" else 3),
                         f"{gap:+.0f}%" if abs(gap) >= 99.5 else "0.0%" if round(gap, 1) == 0 else f"{gap:+.1f}%", p_text(r[f"{loss}_p_runs"]), p_text(r[f"{loss}_p_time"]), WORD[word]]
                for (_, x, ha), text, w, c in zip(sub, cells, ("normal", weight, "normal", "normal", weight),
                                                  (INK, INK, INK, INK, INK if weight == "bold" else MUTED)):
                    ax.text(x0 + half * x, y, text, fontsize=9, color=c, va="center", ha=ha, fontweight=w)
    ax.plot([0, 1], [y - 0.55, y - 0.55], color=INK, lw=0.8)
    fig_.suptitle("Our method against the other methods: statistical tests on five runs", x=0.012, y=0.985, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig_.text(0.012, 0.012,
              "Next-bar log range, 15 cryptocurrencies, 4-hour bars, four test blocks (22 Apr 2025 - 14 Oct 2026, 48,600 forecasts), five runs (training seeds 0-4) per model. "
              "Ours vs. it: mean loss of ours against the mean loss of the other method (negative = ours lower).\n"
              "p, runs: Welch's t-test on the losses of the five runs of each method (is the difference larger than what the training seed moves?). "
              "p, time: Diebold-Mariano test on the loss difference of every bar, averaged over runs and coins\n(is it larger than what the test period moves?). "
              "Result: ours better or worse only if both tests give p < 0.05; runs only / time only: one test alone. HAR and seasonal HAR have one value (one-sample t-test).\n"
              "† The mean QLIKE of these models is carried by a few bars with far too low forecasts.", fontsize=7.2, color=MUTED, ha="left", linespacing=1.4)
    fig_.subplots_adjust(left=0.012, right=0.988, top=0.935, bottom=0.105)
    fig_.savefig(FIG, dpi=220, facecolor="white")
    plt.close(fig_)
    print("written", FIG)


if __name__ == "__main__":
    {"fig": fig}.get((sys.argv[1:] or [""])[0], main)()
