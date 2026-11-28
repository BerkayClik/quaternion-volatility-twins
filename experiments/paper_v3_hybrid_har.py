"""
Hybrids of candidate A and HAR (asked for by the advisor, 2026-11-18: the combination of the quaternion
network with HAR, and the MSEs). The plan mirrors experiments/paper_v3_hybrid.py, with HAR in the
place of LightGBM. Fixed before any fitted combination is computed (the two plain averages had been computed
a few minutes earlier, in the table of MSEs for the advisor; they have nothing to fit):

  partners           HAR (three lags of the log range) and HAR-S x (HAR with one dummy per cell of the week),
                     OLS per coin on the training block, the benchmarks of the paper.
  blend_avg_<p>      (A + partner) / 2
  blend_w_<p>        w * A + (1 - w) * partner, w from a grid of 101 values, least MSE on the validation block
  stack_lin_<p>      OLS of the target on (1, A, partner), fitted on the validation block over all coins
  cascade_res_<p>    A first; then, per coin, an OLS of its residual on the regressors of the partner, fitted
                     on the validation block
  cascade_<p>        the partner first; then candidate A trained on its residual over the whole training
                     block (experiments/paper_v3_cascade.py, configs/paper/v3/a_cascade_<p>_pool_4h.yaml;
                     new training). cascade_<p>_real: the real twin as the second stage.
  blend_avg_real_<p> the average with the real twin of A in the place of A
  blend3, stack_lin3 A, the lagged LightGBM-S and HAR-S x: the plain average of the three, and the OLS stack
                     fitted on the validation block. Does HAR add to the hybrid with LightGBM?

Every combination is fitted per fold on the validation block and applied to the test block; three-seed averages
of the networks. Each row is compared with candidate A alone and with the best hybrid so far, the plain average
of A and the lagged LightGBM-S (0.814 x HAR). Periods: the four test blocks, the 33 days after the sample, all
five blocks. Both periods had been seen for the base models, so this is post hoc.

    .venv/bin/python -W ignore experiments/paper_v3_hybrid_har.py      tables/v3_hybrid_har.csv (with the MSE in the
                                                                       units of the target), v3_hybrid_har_weights.csv
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_hybrid as H  # noqa: E402

OUT = "docs/paper/manuscript_v2/tables"
PARTNERS = {"har": ("HAR", "HAR"), "harsx": ("H", "HAR-S x")}        # name -> (column of H.base, label)
FITTED = ["blend_avg", "blend_w", "stack_lin", "cascade_res", "blend_avg_real"]
BEST = ("HY", "blend_avg_lgbm")                                       # the plain average of A and the lagged LightGBM-S


def _ns(t) -> np.ndarray:
    """Epoch nanoseconds of a time index or column, with or without a time zone."""
    t = pd.DatetimeIndex(t)
    return (t if t.tz is None else t.tz_convert(None)).as_unit("ns").asi8


def regressors(frames: dict) -> dict:
    """dataset -> (epoch ns of its rows, HAR regressors, HAR-S x regressors), as the benchmarks build them."""
    from paper_round7_checks import _interactions
    from src.paper.baselines import HAR_LAGS, _har_design
    out = {}
    for ds, (d, _) in frames.items():
        har = _har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[H.FREQ])
        out[ds] = (_ns(d.frame.index), har, np.column_stack([har, _interactions(d.frame.index, d.step)]))
    return out


def combine(w: pd.DataFrame, reg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """All fitted combinations of one period, in the long format of the analysis. w: base forecasts (H.base)."""
    from src.paper.baselines import _ols
    out, weights = [], []
    for fold, g in w.groupby("fold"):
        va, te = g[g.split == "val"].sort_values(["dataset", "time"]), g[g.split == "test"].sort_values(["dataset", "time"])
        pred, rec = {}, dict(fold=int(fold))
        for name, (col, _) in PARTNERS.items():
            pred[f"blend_avg_{name}"] = (te.N + te[col]) / 2
            pred[f"blend_avg_real_{name}"] = (te.R + te[col]) / 2
            grid = np.linspace(0, 1, 101)
            wbest = float(grid[np.argmin([((a * va.N + (1 - a) * va[col] - va.y) ** 2).mean() for a in grid])])
            pred[f"blend_w_{name}"] = wbest * te.N + (1 - wbest) * te[col]
            beta = np.linalg.lstsq(np.column_stack([np.ones(len(va)), va.N, va[col]]), va.y.to_numpy(), rcond=None)[0]
            pred[f"stack_lin_{name}"] = beta[0] + beta[1] * te.N + beta[2] * te[col]
            rec.update({f"blend_w_{name}_weight_on_network": wbest, f"stack_lin_{name}_intercept": beta[0],
                        f"stack_lin_{name}_network": beta[1], f"stack_lin_{name}_partner": beta[2]})
            second = pd.Series(np.nan, index=te.index)        # per coin: OLS of the residual of A on the partner's regressors
            for ds, v in va.groupby("dataset"):
                t_ns, har, harsx = reg[ds]
                D = har if name == "har" else harsx
                row = pd.Series(np.arange(len(t_ns)), index=t_ns)
                b = _ols(D[row.loc[_ns(v.time)].to_numpy()], (v.y - v.N).to_numpy())
                tds = te[te.dataset == ds]
                second.loc[tds.index] = b[0] + D[row.loc[_ns(tds.time)].to_numpy()] @ b[1:]
            assert second.notna().all()
            pred[f"cascade_res_{name}"] = te.N + second
        pred["blend3"] = (te.N + te.G + te.H) / 3
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(va)), va.N, va.G, va.H]), va.y.to_numpy(), rcond=None)[0]
        pred["stack_lin3"] = beta[0] + beta[1] * te.N + beta[2] * te.G + beta[3] * te.H
        rec.update(stack_lin3_intercept=beta[0], stack_lin3_network=beta[1], stack_lin3_lgbm=beta[2], stack_lin3_harsx=beta[3])
        pred[BEST[1]] = (te.N + te.G) / 2
        weights.append(rec)
        long = lambda p, key: pd.DataFrame({"dataset": te.dataset.to_numpy(), "task": "range", "groupset": key[0], "model": key[1], "seed": 0,
                                            "fold": int(fold), "time": te.time.to_numpy(), "y": te.y.to_numpy(), "pred": np.asarray(p)})
        out += [long(p, ("HY", name)) for name, p in pred.items()]
        out += [long(te[short], key) for short, key in H.KEYS.items()]
    return pd.concat(out, ignore_index=True), weights


def trained(pattern: str, n_folds: int) -> list[pd.DataFrame]:
    """Forecasts of the cascades with a trained second stage (one run per partner), if the runs are complete."""
    import paper_v2_summary as S
    parts = []
    for name in PARTNERS:
        dirs = glob.glob(pattern.format(name))
        if len(dirs) != 1 or not glob.glob(os.path.join(dirs[0], "preds", "*.npz")):
            print("not available: trained cascade with", name)
            continue
        c = S.run(dirs[0])
        for suffix, model in (("", "hq_gattn-pool"), ("_real", "hl_gattn-pool")):
            x = c[(c.groupset == "PRMC") & (c.model == model)]
            if len(x[["dataset", "seed", "fold"]].drop_duplicates()) != 15 * 3 * n_folds:
                print("not complete: trained cascade with", name, suffix)
                continue
            parts.append(x.assign(groupset="HY", model=f"cascade_{name}{suffix}"))
    return parts


def table(df: pd.DataFrame, period: str) -> pd.DataFrame:
    import paper_v2_round2 as R
    st = R.Store([df])
    har = df[(df.groupset == "P") & (df.model == "har")]
    mse_har = float(((har.pred - har.y) ** 2).mean())                 # all coins and test bars (the panel is balanced)
    names = [("candidate A (H-QLSTM)", H.KEYS["N"]), ("real twin (H-LSTM)", H.KEYS["R"]), ("HAR", H.KEYS["HAR"]), ("HAR-S x", H.KEYS["H"]),
             ("LightGBM-S, lagged", H.KEYS["G"]), ("average of A and LightGBM", BEST)]
    for name in PARTNERS:
        names += [(f"{c}_{name}", ("HY", f"{c}_{name}")) for c in FITTED] + [(f"cascade_{name}", ("HY", f"cascade_{name}")),
                                                                              (f"cascade_{name}_real", ("HY", f"cascade_{name}_real"))]
    names += [("blend3", ("HY", "blend3")), ("stack_lin3", ("HY", "stack_lin3"))]
    rows = []
    for label, key in names:
        if not st.has(key):
            continue
        r = dict(period=period, model=label, mse_har=st.vs_har(key))
        r["mse"] = r["mse_har"] * mse_har
        for short, other in (("A", H.KEYS["N"]), ("best", BEST)):
            if key != other:
                v = st.pair(key, other)
                r.update({f"vs_{short}": v["ratio"], f"vs_{short}_lo": v["ci_lo"], f"vs_{short}_hi": v["ci_hi"], f"vs_{short}_better": v["better"], f"p_{short}": v["p"]})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    from paper_round7_checks import _frames
    import paper_v3_confirm as C
    fr4, _ = _frames(H.FREQ)
    t4, w4 = combine(H.base(H.TEST_RUNS), regressors(fr4))
    t4 = pd.concat([t4] + trained("results/v3*/out/v3_a_cascade_{}_pool_4h_*", 4), ignore_index=True)
    frc = {ds: (d, [fold]) for ds, (d, fold) in C.frames().items()}
    t5, w5 = combine(H.base(H.CONFIRM_RUNS), regressors(frc))
    t5 = pd.concat([t5] + trained("results/v3confirm/out/v3_confirm_cascade_{}_pool_4h_*", 1), ignore_index=True)
    assert t5.time.min() > C.OLD_END
    both = pd.concat([t4, t5.assign(fold=4)], ignore_index=True)
    both = both[both.model.isin(set(t4.model) & set(t5.model))]
    t = pd.concat([table(t4, "four test blocks"), table(t5, "33 days after the sample"), table(both, "all five blocks")], ignore_index=True)
    t.to_csv(os.path.join(OUT, "v3_hybrid_har.csv"), index=False)
    wt = pd.concat([pd.DataFrame(w4).assign(period="four test blocks"), pd.DataFrame(w5).assign(period="33 days after the sample")])
    wt.to_csv(os.path.join(OUT, "v3_hybrid_har_weights.csv"), index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200)
    print(t.round(4).to_string(index=False))
    print(wt.round(3).T.to_string())


FIG = "docs/paper/review/advisor_2026-11-17/candidate_a/har_combination_results.png"
FIG_GROUPS = [   # (heading, [(label, row of tables/v3_hybrid_har.csv)])
    ("Single models", [("HAR", "HAR"), ("Seasonal HAR (week cells)", "HAR-S x"), ("LightGBM-S, lagged features", "LightGBM-S, lagged"),
                       ("H-LSTM (real twin)", "real twin (H-LSTM)"), ("H-QLSTM (candidate A)", "candidate A (H-QLSTM)")]),
    ("H-QLSTM combined with HAR", [("Blending: average", "blend_avg_har"), ("Blending: weighted average", "blend_w_har"),
                                   ("Stacking: linear", "stack_lin_har"), ("Cascade: HAR, then H-QLSTM on its residual", "cascade_har"),
                                   ("Cascade: H-QLSTM, then HAR on its residual", "cascade_res_har")]),
    ("H-QLSTM combined with seasonal HAR", [("Blending: average", "blend_avg_harsx"), ("Blending: weighted average", "blend_w_harsx"),
                                            ("Stacking: linear", "stack_lin_harsx"), ("Cascade: seasonal HAR, then H-QLSTM", "cascade_harsx"),
                                            ("Cascade: H-QLSTM, then seasonal HAR", "cascade_res_harsx")]),
    ("H-QLSTM combined with LightGBM", [("Blending: average", "average of A and LightGBM")]),
]


def fig():
    """Chart for the advisor from tables/v3_hybrid_har.csv (four test blocks): MSE below that of HAR of the single
    models and of every combination of the H-QLSTM with HAR, with the MSE printed. One accent colour for the best
    row, a second neutral for the H-QLSTM alone, whose value is also drawn as a reference line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.read_csv(os.path.join(OUT, "v3_hybrid_har.csv")).query("period == 'four test blocks'").set_index("model")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    ACCENT, GREY, DARK, INK, MUTED = "#0072B2", "#A6A6A6", "#5E5E5E", "#1a1a1a", "#6b6b6b"
    n_rows = sum(len(rows) for _, rows in FIG_GROUPS) + len(FIG_GROUPS)
    fig_, ax = plt.subplots(figsize=(9.6, 0.33 * n_rows + 1.25))
    alone = 100 * (1 - float(t.loc["candidate A (H-QLSTM)", "mse_har"]))
    y, ticks, labels = n_rows, [], []
    for heading, rows in FIG_GROUPS:
        y -= 1
        ax.text(-0.35, y - 0.05, heading, ha="right", va="center", fontsize=9, fontweight="bold", color=INK)
        for label, name in rows:
            y -= 1
            gain, mse = 100 * (1 - float(t.loc[name, "mse_har"])), float(t.loc[name, "mse"])
            best, own = name == "average of A and LightGBM", name == "candidate A (H-QLSTM)"
            ax.barh(y, gain, height=0.64, color=ACCENT if best else DARK if own else GREY, zorder=2)
            ax.text(gain + 0.25, y, f"{gain:.1f}%", va="center", ha="left", fontsize=9, color=INK, fontweight="bold" if best or own else "normal",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=4)      # the reference line must not cross a value
            ax.text(23.2, y, f"{mse:.4f}", va="center", ha="right", fontsize=9, color=INK, fontweight="bold" if best or own else "normal")
            ticks.append(y); labels.append(label)
    ax.plot([alone, alone], [-0.6, n_rows - 1.45], color=DARK, lw=0.9, ls=(0, (4, 3)), zorder=3)
    ax.text(alone, n_rows - 1.05, "H-QLSTM alone", ha="center", va="center", fontsize=8, color=DARK)
    ax.text(23.2, n_rows - 1.05, "MSE", ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(-0.6, n_rows - 0.5)
    ax.set_xlim(0, 23.5)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.xaxis.grid(True, color="0.90", lw=0.6, zorder=0)
    ax.tick_params(axis="x", length=0, labelsize=8, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_color("0.75")
    ax.set_xlabel("MSE below that of HAR (%), higher is better", fontsize=8.5, color=MUTED)
    fig_.suptitle("H-QLSTM combined with HAR: error of the next-bar log-range forecast",
                  x=0.012, y=0.985, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig_.text(0.012, 0.012, "15 cryptocurrencies, 4-hour bars, four test blocks (22 Apr 2025 - 14 Oct 2026, 48,600 forecasts), three-seed averages. Blending, stacking and the cascades "
              "that end with HAR\nare fitted on the validation block of each fold; in the cascades that start with HAR the H-QLSTM is trained on the residual over the whole training block.",
              fontsize=7.3, color=MUTED, ha="left", linespacing=1.35)
    fig_.tight_layout(rect=(0, 0.055, 1, 0.975))
    fig_.savefig(FIG, dpi=220, facecolor="white")
    plt.close(fig_)
    print("written", FIG)


if __name__ == "__main__":
    fig() if sys.argv[1:] == ["fig"] else main()
