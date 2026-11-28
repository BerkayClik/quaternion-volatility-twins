"""
Hybrids of the hierarchical quaternion LSTM (candidate A) and gradient boosting: blending, stacking and
cascading. Asked for by the advisor on 2026-11-18.

This plan was committed before any combination was computed. The forecasts of the base models were known
(docs/FINDINGS.md: on the four test blocks A 0.821 and the lagged LightGBM-S 0.828 times the MSE of HAR; on
the block after the paper's sample 0.835 and 0.817), so every result of this file is post hoc for both
periods and needs bars after 2026-11-16 to be confirmed.

BASE FORECASTS (three-seed averages; the saved validation and test forecasts of every fold)
  N   candidate A, hq_gattn on price with range, market, calendar
  R   its real twin, hl_gattn on the same inputs
  G   pooled LightGBM-S with lagged exchange features
  G0  pooled LightGBM-S on the exchange features of the current bar
  H   HAR with one dummy per cell of the week

COMBINATIONS. Each is fitted, fold by fold, on the validation block of that fold only (the 180 days before
the test block; the base models used that block for early stopping and nothing later) with one fit for all
coins, and is applied to the test block of the fold.
  blend_avg      (N + G) / 2, no fitted parameter
  blend_w        w N + (1 - w) G with w in [0, 1] that minimises the validation MSE
  stack_lin      OLS of the target on N and G with an intercept
  stack_lin4     OLS of the target on N, G, G0 and H with an intercept
  stack_gbm      LightGBM with small trees (7 leaves, 200 trees) on N, G, G0, H and the calendar cell of
                 the target bar
  cascade_res    a second stage after the network: LightGBM on the regressors of the lagged LightGBM-S,
                 trained on the residual y - N (first 135 days of the validation block; last 45 days for
                 early stopping). Forecast: N plus the second stage.
  cascade_feat   LightGBM on the same regressors plus N as a further regressor (same split of the block).
  blend_avg_real, stack_lin_real   blend_avg and stack_lin with R in place of N: does the algebra matter
                 inside a hybrid?

EVALUATION as in the paper: MSE over all coins and bars relative to HAR, ratio of every combination to N and
to G with the 90% interval and the cross-coin DM test, (a) on the four test blocks, (b) on the 33 days after
the paper's sample (the fold of paper_v3_confirm.py) and, added after the first run, on the five blocks
together. All combinations are reported, not the best one.
A cascade in the other direction (the network trained on the residual of LightGBM) needs new training
and is not part of this file.

    .venv/bin/python -W ignore experiments/paper_v3_hybrid.py        tables/v3_hybrid.csv, v3_hybrid_weights.csv
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

OUT = "docs/paper/manuscript_v2/tables"
FREQ, WINDOW = "4h", 42
KEYS = {"N": ("PRMC", "hq_gattn-pool"), "R": ("PRMC", "hl_gattn-pool"), "G": ("PM", "lgbmsl-pool"),
        "G0": ("PM", "lgbms-pool"), "H": ("P", "hars_x"), "HAR": ("P", "har")}
TEST_RUNS = ["results/final_fa/out/final_single_4h_*", "results/v3local/v3_gattn_range_pool_4h_*",
             "results/r8_local/final_lgbmlags_4h_r8", "results/r8_local/final_harsx_4h_r8",
             "results/final_local/final_lgbm_4h_*", "results/final_local/final_lgbm_seeds_4h_*"]
CONFIRM_RUNS = ["results/v3confirm/bench", "results/v3confirm/out/v3_confirm_gattn_range_pool_4h_*"]
COMBOS = ["blend_avg", "blend_w", "stack_lin", "stack_lin4", "stack_gbm", "cascade_res", "cascade_feat",
          "blend_avg_real", "stack_lin_real"]


def base(patterns: list[str]) -> pd.DataFrame:
    """Wide frame of the base forecasts: one row per (dataset, fold, split, time) with y and one column per model."""
    import paper_v2_round2 as R
    ens = R._splits(list(KEYS.values()), patterns)
    name = {v: k for k, v in KEYS.items()}
    ens["m"] = [name[(g, m)] for g, m in zip(ens.groupset, ens.model)]
    wide = ens.pivot_table(index=["dataset", "fold", "split", "time"], columns="m", values="pred")
    wide["y"] = ens.groupby(["dataset", "fold", "split", "time"]).y.first()
    return wide.dropna().reset_index()


def designs(frames: dict) -> pd.DataFrame:
    """Regressors of the lagged LightGBM-S for every row, standardised per coin with the statistics of the
    training block, as predict_baseline_pooled builds them. frames: dataset -> (Dataset, [folds])."""
    from src.paper.baselines import HAR_LAGS, _design
    parts = []
    for ds, (d, folds) in frames.items():
        X = d.X(["price", "market"])
        Xl = np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS[FREQ][1:]])
        D = _design("lgbms", d.frame, FREQ, Xl)
        y = d.frame["y_range"].to_numpy(float)
        idx = d.frame.index
        for k, fold in enumerate(folds):
            ok = fold["train"][~np.isnan(D[fold["train"]]).any(1)]
            mx, sx = D[ok].mean(0), D[ok].std(0)
            sx[sx < 1e-8] = 1.0
            rows = np.concatenate([r[r >= WINDOW - 1] for r in (fold["val"], fold["test"])])
            f = pd.DataFrame((D[rows] - mx) / sx, columns=[f"x{i}" for i in range(D.shape[1])])
            f.insert(0, "sy", y[ok].std())
            f.insert(0, "time", idx[rows])
            f.insert(0, "fold", k)
            f.insert(0, "dataset", ds)
            parts.append(f)
    return pd.concat(parts, ignore_index=True)


def _cell(time: pd.Series) -> np.ndarray:
    nxt = time + pd.Timedelta("4h")
    return (nxt.dt.dayofweek * 6 + nxt.dt.hour // 4).to_numpy()


def combine(w: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """All combinations for one period. w: base forecasts (base); X: regressors (designs)."""
    import lightgbm as lgb
    from src.paper.baselines import LGBM_PARAMS
    w = w.merge(X, on=["dataset", "fold", "time"], how="left")
    xcols = [c for c in w.columns if c.startswith("x") and c[1:].isdigit()]
    assert not w[xcols].isna().any().any(), "regressors missing for some rows"
    out, weights = [], []
    for fold, g in w.groupby("fold"):
        va, te = g[g.split == "val"].sort_values("time"), g[g.split == "test"].copy()
        pred = {}
        pred["blend_avg"] = (te.N + te.G) / 2
        pred["blend_avg_real"] = (te.R + te.G) / 2
        grid = np.linspace(0, 1, 101)
        wbest = float(grid[np.argmin([((a * va.N + (1 - a) * va.G - va.y) ** 2).mean() for a in grid])])
        pred["blend_w"] = wbest * te.N + (1 - wbest) * te.G
        rec = dict(fold=int(fold), blend_w_weight_on_network=wbest)
        for name, cols in (("stack_lin", ["N", "G"]), ("stack_lin_real", ["R", "G"]), ("stack_lin4", ["N", "G", "G0", "H"])):
            A = np.column_stack([np.ones(len(va))] + [va[c] for c in cols])
            beta = np.linalg.lstsq(A, va.y.to_numpy(), rcond=None)[0]
            pred[name] = beta[0] + sum(b * te[c] for b, c in zip(beta[1:], cols))
            rec.update({f"{name}_intercept": beta[0], **{f"{name}_{c}": b for b, c in zip(beta[1:], cols)}})
        meta = lambda d: np.column_stack([d.N, d.G, d.G0, d.H, _cell(d.time)])
        m = lgb.LGBMRegressor(objective="regression", learning_rate=0.03, num_leaves=7, min_child_samples=200,
                              n_estimators=200, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=4)
        m.fit(meta(va), va.y.to_numpy())
        pred["stack_gbm"] = m.predict(meta(te))
        # second stages: first 135 days of the validation block to fit, the last 45 days for early stopping
        cut = va.time.min() + pd.Timedelta(days=135)
        fit, stop = va[va.time <= cut], va[va.time > cut]
        res = lambda d: ((d.y - d.N) / d.sy).to_numpy()                 # residual of the network in the coin's units
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(fit[xcols].to_numpy(), res(fit), eval_set=[(stop[xcols].to_numpy(), res(stop))], callbacks=[lgb.early_stopping(100, verbose=False)])
        pred["cascade_res"] = te.N + te.sy * m.predict(te[xcols].to_numpy())
        rec["cascade_res_trees"] = int(m.best_iteration_ or 0)
        zf = lambda d: np.column_stack([d[xcols].to_numpy(), (d.N / d.sy).to_numpy()])
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(zf(fit), (fit.y / fit.sy).to_numpy(), eval_set=[(zf(stop), (stop.y / stop.sy).to_numpy())], callbacks=[lgb.early_stopping(100, verbose=False)])
        pred["cascade_feat"] = te.sy * m.predict(zf(te))
        rec["cascade_feat_trees"] = int(m.best_iteration_ or 0)
        weights.append(rec)
        for name in COMBOS:
            out.append(pd.DataFrame({"dataset": te.dataset.to_numpy(), "task": "range", "groupset": "HY", "model": name, "seed": 0,
                                     "fold": int(fold), "time": te.time.to_numpy(), "y": te.y.to_numpy(), "pred": np.asarray(pred[name])}))
        for short, key in KEYS.items():
            out.append(pd.DataFrame({"dataset": te.dataset.to_numpy(), "task": "range", "groupset": key[0], "model": key[1], "seed": 0,
                                     "fold": int(fold), "time": te.time.to_numpy(), "y": te.y.to_numpy(), "pred": te[short].to_numpy()}))
    return pd.concat(out, ignore_index=True), weights


def table(df: pd.DataFrame, period: str) -> pd.DataFrame:
    import paper_v2_round2 as R
    st = R.Store([df])
    rows = []
    names = [("candidate A (N)", KEYS["N"]), ("real twin (R)", KEYS["R"]), ("LightGBM-S, lagged (G)", KEYS["G"]),
             ("LightGBM-S (G0)", KEYS["G0"]), ("HAR-S x (H)", KEYS["H"])] + [(c, ("HY", c)) for c in COMBOS]
    for label, key in names:
        r = dict(period=period, model=label, mse_har=st.vs_har(key))
        for other in ("N", "G"):
            if key != KEYS[other]:
                v = st.pair(key, KEYS[other])
                r.update({f"vs_{other}": v["ratio"], f"vs_{other}_lo": v["ci_lo"], f"vs_{other}_hi": v["ci_hi"],
                          f"vs_{other}_better": v["better"], f"p_{other}": v["p"]})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    from paper_round7_checks import _frames
    import paper_v3_confirm as C
    fr4, _ = _frames(FREQ)
    t4, w4 = combine(base(TEST_RUNS), designs(fr4))
    frc = {ds: (d, [fold]) for ds, (d, fold) in C.frames().items()}
    t5, w5 = combine(base(CONFIRM_RUNS), designs(frc))
    assert t5.time.min() > C.OLD_END
    both = pd.concat([t4, t5.assign(fold=4)], ignore_index=True)
    t = pd.concat([table(t4, "four test blocks"), table(t5, "33 days after the sample"), table(both, "all five blocks")],
                  ignore_index=True)
    t.to_csv(os.path.join(OUT, "v3_hybrid.csv"), index=False)
    wt = pd.concat([pd.DataFrame(w4).assign(period="four test blocks"), pd.DataFrame(w5).assign(period="33 days after the sample")])
    wt.to_csv(os.path.join(OUT, "v3_hybrid_weights.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(t.round(4).to_string(index=False))
    print(wt.round(3).to_string(index=False))


FIG = "docs/paper/review/advisor_2026-11-17/candidate_a/hybrid_results.png"
CASCADE_NET = "cascade: LightGBM, then H-QLSTM on the residual"     # row of tables/v3_cascade.csv (paper_v3_cascade.py)
FIG_ROWS = [("Blending: average of H-QLSTM and LightGBM", "blend_avg"), ("Blending: weighted average", "blend_w"),
            ("Stacking: linear", "stack_lin"), ("Cascade: LightGBM, then H-QLSTM on its residual", CASCADE_NET),
            ("Cascade: H-QLSTM, then LightGBM on its residual", "cascade_res"),
            ("Candidate A (H-QLSTM) alone", "candidate A (N)"), ("LightGBM-S, lagged features, alone", "LightGBM-S, lagged (G)"),
            ("Stacking: LightGBM as the combiner", "stack_gbm"), ("Seasonal HAR (week cells)", "HAR-S x (H)")]


def fig():
    """Chart for the advisor from tables/v3_hybrid.csv: MSE below that of HAR (per cent) of the hybrids and of the
    two base models, on the four test blocks and on the 33 days after the sample. One accent colour for the
    simple average, neutral grey for the rest; the names and the printed values carry the identity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.concat([pd.read_csv(os.path.join(OUT, "v3_hybrid.csv")),
                   pd.read_csv(os.path.join(OUT, "v3_cascade.csv")).query("model == @CASCADE_NET")]).set_index(["period", "model"]).mse_har
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    ACCENT, GREY, INK, MUTED = "#0072B2", "#8C8C8C", "#1a1a1a", "#6b6b6b"
    fig_, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    panels = [("four test blocks", "Four test blocks of the manuscript\n22 Apr 2025 - 14 Oct 2026 (540 days)"),
              ("33 days after the sample", "After the sample\n15 Oct - 16 Nov 2026 (33 days)")]
    y = list(range(len(FIG_ROWS)))[::-1]
    for ax, (period, title) in zip(axes, panels):
        for yi, (label, name) in zip(y, FIG_ROWS):
            gain = 100 * (1 - float(t[(period, name)]))
            first = name == "blend_avg"
            ax.barh(yi, gain, height=0.62, color=ACCENT if first else GREY, zorder=2)
            ax.text(gain + 0.25, yi, f"{gain:.1f}%", va="center", ha="left", fontsize=9, color=INK, fontweight="bold" if first else "normal")
        ax.set_title(title, fontsize=9.5, loc="left", color=INK, pad=8)
        ax.set_xlim(0, 21.5)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.xaxis.grid(True, color="0.90", lw=0.6, zorder=0)
        ax.tick_params(axis="x", length=0, labelsize=8, colors=MUTED)
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_color("0.75")
        ax.set_xlabel("MSE below that of HAR (%), higher is better", fontsize=8.5, color=MUTED)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in FIG_ROWS], fontsize=9, color=INK)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    fig_.suptitle("Blending, stacking and cascading of the H-QLSTM and LightGBM: error of the next-bar log-range forecast",
                  x=0.012, y=0.975, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig_.text(0.012, 0.012, "15 cryptocurrencies, 4-hour bars, three-seed averages. Blending, stacking and the second cascade are fitted on the validation block "
              "of each fold;\nin the first cascade the H-QLSTM is trained on the residual of LightGBM over the whole training block. "
              "Both periods had been seen for the two base models.", fontsize=7.5, color=MUTED, ha="left", linespacing=1.35)
    fig_.tight_layout(rect=(0, 0.065, 1, 0.985), w_pad=1.6)
    fig_.savefig(FIG, dpi=220, facecolor="white")
    plt.close(fig_)
    print("written", FIG)


if __name__ == "__main__":
    fig() if sys.argv[1:] == ["fig"] else main()
