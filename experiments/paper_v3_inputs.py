"""
Effect of the data types on the forecast (asked for by the advisor, 2026-11-20: a table that measures the
effect of the different data types). Tables exist for parts of the question: tables/v2_sources.csv for the model of
paper v2, and the rows "Inputs" of tables/v3_grid.csv for candidate A (three seeds, averages of the seeds: without
the calendar group 1.052 of candidate A, without the market group 1.008, with the social and sentiment groups 1.001,
and 0.992 for the real twin). This file completes them into one table for the current system and its parts.
Fixed before any of the new runs was started.

  data types   price      high, low and close relative to the previous close, and the log range of the bar
               market     surprise of the quote volume and of the number of trades, taker imbalance, change of volume
               calendar   time of day and day of week of the target bar
               technical  RSI, MACD histogram, NATR, Bollinger %b (TA-Lib on exchange prices)
               social     active and new contributors, active and new posts (vendor)
               sentiment  sentiment, Galaxy Score, social dominance, interactions (vendor)
  input sets   reference: price, market, calendar (the inputs of candidate A and of the LightGBM benchmark).
               One type removed: market; calendar. Added to the reference: technical; social; sentiment; social and
               sentiment; technical, social and sentiment (all six types). Price is in every set.
  models       N      the H-QLSTM in the architecture of candidate A (hq_gattn: one quaternion LSTM per data type,
                      attention over their summaries), 20,000 parameters for every input set, so that the encoders
                      narrow as types are added (hidden size 21, 17, 15, 13, 12 quaternions for 2 to 6 types)
               R      its real twin (hl_gattn)
               G      the pooled LightGBM-S with lagged features, the benchmark of the paper: the HAR regressors of
                      the log range, the features of the data types of the current bar and their means over the HAR
                      lags, and dummies for the calendar of the target bar (left out in "without calendar")
               N + G  the plain average of N and G on the same input set, the best hybrid so far
  run          k = 0..4 as in experiments/paper_v3_runs.py; run k of the average uses run k of N and of G. Exist:
               the reference (five runs of N, R and G) and seeds 0-2 of N and R without market, without calendar and
               with social and sentiment. New: seeds 3-4 of these (configs/paper/v3/a_inputs_seeds_pool_4h.yaml),
               five seeds of the other four sets (a_types_ts_pool_4h.yaml, a_types_ea_pool_4h.yaml), and G on the
               seven sets other than the reference (`lgbm` below).
  reported     as in paper_v3_runs: mean and standard deviation over the five runs of MSE and QLIKE on the four test
               blocks, in the units of the loss and relative to HAR. For the average of the five runs' forecasts:
               the MSE over that of the same model on the reference inputs, with the 90% interval and the cross-coin
               DM test (p adjusted by Holm over the seven sets of a model in the CSV file); for N also the MSE over
               that of R on the same inputs.
  rule         A data type counts as useful for a model only if adding it lowers the MSE (ratio below one, p < 0.05)
               or removing it raises the MSE (ratio above one, p < 0.05). Every row is reported.
  not done     the 33 days after the sample (the vendor data end with the sample); other bar lengths; input sets
               without the range of the bar (a matter of how the price is written, rows "Inputs" of v3_grid.csv).

    .venv/bin/python -W ignore experiments/paper_v3_inputs.py check    the LightGBM code below reproduces the benchmark
    .venv/bin/python -W ignore experiments/paper_v3_inputs.py lgbm     G on the seven new input sets (results/v3runs)
    .venv/bin/python -W ignore experiments/paper_v3_inputs.py          tables/v3_inputs.csv
    .venv/bin/python -W ignore experiments/paper_v3_inputs.py fig      review/advisor_2026-11-17/candidate_a/data_types_table.png
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import paper_v3_runs as RU  # noqa: E402

OUT, RUNS, V3, FREQ = RU.OUT, RU.RUNS, RU.V3, "4h"
LGBM_DIR = "results/v3runs/v3_inputs_lgbm_4h"
LETTER = {"P": "price", "M": "market", "T": "technical", "S": "social", "E": "sentiment"}
REF = "price, market, calendar"
SETS = [   # (input set, groupset of the networks, (groupset, model) of LightGBM)
    (REF, "PRMC", ("PM", "lgbmsl-pool")),
    ("without market", "PRC", ("P", "lgbmsl-pool")),
    ("without calendar", "PRM", ("PM", "lgbml-pool")),
    ("+ technical", "PRMTC", ("PMT", "lgbmsl-pool")),
    ("+ social", "PRMSC", ("PMS", "lgbmsl-pool")),
    ("+ sentiment", "PRMEC", ("PME", "lgbmsl-pool")),
    ("+ social, sentiment", "PRMSEC", ("PMSE", "lgbmsl-pool")),
    ("+ technical, social, sentiment", "PRMTSEC", ("PMTSE", "lgbmsl-pool")),
]
INPUT_RUNS = [V3.format("inputs_pool_4h"), V3.format("inputs_seeds_pool_4h")]
NET_RUNS = {"PRMC": RU.A_RUNS, "PRC": INPUT_RUNS, "PRM": INPUT_RUNS, "PRMSEC": INPUT_RUNS,
            "PRMTC": [V3.format("types_ts_pool_4h")], "PRMSC": [V3.format("types_ts_pool_4h")],
            "PRMEC": [V3.format("types_ea_pool_4h")], "PRMTSEC": [V3.format("types_ea_pool_4h")]}
LGBM_REF = RU.SEEDED["LightGBM-S, lagged"][1]
MODELS = ["LightGBM", "H-LSTM (real twin)", "H-QLSTM", "H-QLSTM + LightGBM"]


def _lgbm_fit(fr: dict, window: int, groups: list[str], calendar: bool, seed: int, fold: int) -> list[dict]:
    """The lagged LightGBM-S of the paper (paper_round8_checks.lgbm_save) on the features of `groups`: the current
    bar and the means over the longer HAR lags; `calendar` False leaves out the dummies of the target bar."""
    from src.paper.baselines import HAR_LAGS, predict_baseline_pooled
    dss = list(fr)
    Xs = [fr[ds][0].X(groups) for ds in dss]
    Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS[FREQ][1:]]) for X in Xs]
    return predict_baseline_pooled("lgbms" if calendar else "lgbm", [fr[ds][0].frame for ds in dss], "range",
                                   [fr[ds][1][fold] for ds in dss], window, FREQ, Xl, seed=seed)


def check():
    """The code above gives the saved forecasts of the benchmark (price and market, seed 0, first fold)."""
    import lightgbm  # noqa: F401  (load its OpenMP runtime first, as the runner does)
    from paper_round7_checks import _frames
    fr, window = _frames(FREQ)
    res = _lgbm_fit(fr, window, ["price", "market"], True, 0, 0)
    gap = 0.0
    for ds, r in zip(fr, res):
        z = np.load(os.path.join(LGBM_REF[0], "preds", f"{ds}__range__PM__lgbmsl-pool__s0__f0.npz"))
        gap = max(gap, float(np.abs(z["test_pred"] - r["test_pred"].astype("float32")).max()))
    print("largest gap to the saved benchmark forecasts:", gap)
    assert gap < 1e-5


def lgbm(seeds=RUNS):
    """LightGBM on the seven input sets other than the reference, saved in the format of the runner."""
    import lightgbm  # noqa: F401
    from paper_round7_checks import _frames
    fr, window = _frames(FREQ)
    dss = list(fr)
    out = os.path.join(LGBM_DIR, "preds")
    os.makedirs(out, exist_ok=True)
    json.dump({"config": {"name": "v3_inputs_lgbm_4h", "note": "paper_v3_inputs.lgbm"}}, open(os.path.join(LGBM_DIR, "config.json"), "w"), indent=1)
    for name, _, (gs, model) in SETS[1:]:
        groups = [LETTER[c] for c in gs]
        for seed in seeds:
            for k in range(len(fr[dss[0]][1])):
                paths = [os.path.join(out, f"{ds}__range__{gs}__{model}__s{seed}__f{k}.npz") for ds in dss]
                if all(os.path.exists(p) for p in paths):
                    continue
                t0 = time.time()
                res = _lgbm_fit(fr, window, groups, model == "lgbmsl-pool", seed, k)
                for ds, r, p in zip(dss, res, paths):
                    frame = fr[ds][0].frame
                    idx = frame.index if frame.index.tz is None else frame.index.tz_convert(None)
                    t = idx.as_unit("ns").asi8                         # epoch ns, as the runner saves
                    y = frame["y_range"].to_numpy(float)
                    np.savez(p, test_time=t[r["test_rows"]], test_y=y[r["test_rows"]].astype("float32"),
                             test_pred=r["test_pred"].astype("float32"), val_time=t[r["val_rows"]],
                             val_y=y[r["val_rows"]].astype("float32"), val_pred=r["val_pred"].astype("float32"),
                             meta=np.array(json.dumps(dict(freq=FREQ, seed=seed, fold=k, model=model, groups=groups))))
                print(name, "seed", seed, "fold", k, f"{time.time() - t0:.0f}s", flush=True)


def main():
    import paper_v2_round2 as R
    from src.paper.evaluate import holm
    caps = R.train_caps()
    har = RU.forecasts(*RU.FIXED["HAR"])[0]
    mse_har, ql_har = RU.losses(har, caps)
    runs = {}
    for name, gs, key in SETS:
        N, T = RU.forecasts((gs, "hq_gattn-pool"), NET_RUNS[gs]), RU.forecasts((gs, "hl_gattn-pool"), NET_RUNS[gs])
        G = RU.forecasts(key, LGBM_REF if name == REF else [LGBM_DIR])
        runs["LightGBM", name], runs["H-LSTM (real twin)", name], runs["H-QLSTM", name] = G, T, N
        runs["H-QLSTM + LightGBM", name] = {k: N[k].assign(pred=(N[k].pred + G[k].reindex(N[k].index).pred) / 2) for k in RUNS if k in N and k in G}
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    rows, ens = [], {}
    for model in MODELS:
        for name, _, _ in SETS:
            have = [k for k in RUNS if k in runs[model, name]]
            if not have:
                print("not available:", model, "|", name)
                continue
            for k in have:
                assert np.allclose(runs[model, name][k].y, har.reindex(runs[model, name][k].index).y, atol=1e-4), f"{model}, {name}: targets differ from those of HAR"
            per = np.array([RU.losses(runs[model, name][k], caps) for k in have])
            first = runs[model, name][have[0]]
            ens[model, name] = first.assign(pred=np.mean([runs[model, name][k].reindex(first.index).pred for k in have], axis=0))
            mse_avg, ql_avg = RU.losses(ens[model, name], caps)
            rows.append(dict(model=model, inputs=name, runs=len(have), mse_mean=per[:, 0].mean(), mse_sd=sd(per[:, 0]),
                             mse_har_mean=(per[:, 0] / mse_har).mean(), mse_har_sd=sd(per[:, 0] / mse_har),
                             qlike_mean=per[:, 1].mean(), qlike_sd=sd(per[:, 1]),
                             qlike_har_mean=(per[:, 1] / ql_har).mean(), qlike_har_sd=sd(per[:, 1] / ql_har),
                             mse_har_avg=mse_avg / mse_har, qlike_har_avg=ql_avg / ql_har))
    t = pd.DataFrame(rows)

    def long(key: tuple) -> pd.DataFrame:
        e = ens[key][ens[key].index.get_level_values("split") == "test"].reset_index()
        return pd.DataFrame({"dataset": e.dataset, "task": "range", "groupset": "IN", "model": " | ".join(key), "seed": 0,
                             "fold": e.fold, "time": e.time, "y": e.y, "pred": e.pred})

    st = R.Store([long(key) for key in ens])
    for i, r in t.iterrows():
        pairs = [("ref", (r.model, REF))] if r.inputs != REF else []
        if r.model == "H-QLSTM":
            pairs.append(("twin", ("H-LSTM (real twin)", r.inputs)))
        for short, other in pairs:
            if other not in ens:
                continue
            v = st.pair(("IN", f"{r.model} | {r.inputs}"), ("IN", " | ".join(other)))
            for col, val in ((f"vs_{short}", v["ratio"]), (f"vs_{short}_lo", v["ci_lo"]), (f"vs_{short}_hi", v["ci_hi"]),
                             (f"vs_{short}_better", v["better"]), (f"p_{short}", v["p"])):
                t.loc[i, col] = val
        if r.inputs != REF and (r.model, REF) in ens:
            t.loc[i, "qlike_vs_ref"] = r.qlike_har_avg / t.loc[(t.model == r.model) & (t.inputs == REF), "qlike_har_avg"].iloc[0]
    if "p_ref" in t.columns:
        for model, g in t[t.p_ref.notna()].groupby("model"):
            t.loc[g.index, "p_ref_holm"] = holm(g.p_ref.to_numpy())
    t.to_csv(os.path.join(OUT, "v3_inputs.csv"), index=False)
    pm = lambda m, s: f"{m:.4f}" + (f" ± {s:.4f}" if np.isfinite(s) else "")
    col = lambda c: t.get(c, pd.Series(np.nan, index=t.index))
    show = pd.DataFrame({"model": t.model, "inputs": t.inputs, "runs": t.runs, "MSE": [pm(m, s) for m, s in zip(t.mse_mean, t.mse_sd)],
                         "MSE / HAR": [pm(m, s) for m, s in zip(t.mse_har_mean, t.mse_har_sd)],
                         "QLIKE": [pm(m, s) for m, s in zip(t.qlike_mean, t.qlike_sd)], "vs ref": col("vs_ref").round(4),
                         "p": col("p_ref").round(4), "Holm": col("p_ref_holm").round(4), "coins": col("vs_ref_better"),
                         "QLIKE vs ref": col("qlike_vs_ref").round(4), "vs twin": col("vs_twin").round(4), "p twin": col("p_twin").round(4)})
    pd.set_option("display.width", 280, "display.max_columns", 20)
    print(show.to_string(index=False))


FIG = "docs/paper/review/advisor_2026-11-17/candidate_a/data_types_table.png"
FIG_ROWS = [   # (label, input set, indent)
    ("Price, market, calendar (the model)", REF, 0),
    ("without market data (price, calendar)", "without market", 1),
    ("without the calendar (price, market)", "without calendar", 1),
    ("+ technical indicators", "+ technical", 1),
    ("+ social", "+ social", 1),
    ("+ sentiment", "+ sentiment", 1),
    ("+ social, sentiment", "+ social, sentiment", 1),
    ("+ technical, social, sentiment (all data)", "+ technical, social, sentiment", 1),
]


def fig():
    """Table for the advisor from tables/v3_inputs.csv, drawn as an image: one row per input set, one pair of columns
    per model (mean ± standard deviation over the runs, and the change against the reference inputs); MSE above,
    QLIKE below. The change is that of the average of the runs' forecasts; the mark of significance is for the MSE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})
    INK, MUTED, RULE, BAND = "#1a1a1a", "#6b6b6b", "#c9c9c9", "#f3f6f9"
    t = pd.read_csv(os.path.join(OUT, "v3_inputs.csv")).set_index(["model", "inputs"])
    left, span = 0.275, (1 - 0.275) / len(MODELS)
    x_val = [left + span * (j + 0.62) for j in range(len(MODELS))]      # right edge of "mean ± sd"
    x_chg = [left + span * (j + 0.955) for j in range(len(MODELS))]     # right edge of the change
    panels = [("MSE", "mse_mean", "mse_sd", 4, "vs_ref"), ("QLIKE", "qlike_mean", "qlike_sd", 3, "qlike_vs_ref")]
    n_rows = len(panels) * (len(FIG_ROWS) + 1) + 1
    fig_, ax = plt.subplots(figsize=(12.6, 0.285 * (n_rows + 1) + 1.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0.7, n_rows + 2.2)
    y = n_rows + 1.6
    ax.text(0.0, y - 0.5, "Inputs", fontsize=9, color=MUTED, va="center")
    for j, model in enumerate(MODELS):
        ax.text(left + span * (j + 0.52), y, model, fontsize=9, fontweight="bold", color=INK, va="center", ha="center")
        ax.plot([left + span * j + 0.012, left + span * (j + 1) - 0.004], [y - 0.5, y - 0.5], color=RULE, lw=0.6)
        ax.text(x_val[j], y - 1, "mean ± sd", fontsize=8.5, color=MUTED, va="center", ha="right")
        ax.text(x_chg[j], y - 1, "change", fontsize=8.5, color=MUTED, va="center", ha="right")
    y -= 1
    ax.plot([0, 1], [y - 0.5, y - 0.5], color=INK, lw=0.8)
    y -= 0.05
    for heading, mean, sd, digits, change in panels:
        y -= 1
        ax.text(0.0, y, heading, fontsize=9, fontweight="bold", color=INK, va="center")
        ax.plot([0, 1], [y - 0.45, y - 0.45], color=RULE, lw=0.6)
        best = {m: t.loc[m][mean].idxmin() for m in MODELS if m in t.index.get_level_values(0)}
        for label, name, indent in FIG_ROWS:
            y -= 1
            if name == REF:
                ax.add_patch(plt.Rectangle((0, y - 0.47), 1, 0.94, color=BAND, zorder=0, lw=0))
            ax.text(0.012 + 0.014 * indent, y, label, fontsize=9, color=INK, va="center")
            for j, model in enumerate(MODELS):
                if (model, name) not in t.index:
                    ax.text(x_val[j], y, "not finished", fontsize=8.5, color=MUTED, va="center", ha="right")
                    continue
                r = t.loc[(model, name)]
                ax.text(x_val[j], y, RU._pm(r[mean], r[sd], digits), fontsize=9, color=INK, va="center", ha="right",
                        fontweight="bold" if best.get(model) == name else "normal")
                if name == REF or not np.isfinite(r.get(change, np.nan)):
                    continue
                gap = round(100 * (r[change] - 1), 1)
                ax.text(x_chg[j] - 0.008, y, "0.0%" if gap == 0 else f"{gap:+.1f}%", fontsize=9, color=INK, va="center", ha="right")
                if change == "vs_ref" and r.p_ref < 0.05:
                    ax.text(x_chg[j] - 0.008, y, " *", fontsize=9, color=INK, va="center", ha="left")
    ax.plot([0, 1], [y - 0.55, y - 0.55], color=INK, lw=0.8)
    fig_.suptitle("Effect of the data types: five runs per model and input set, mean ± standard deviation", x=0.012, y=0.985, ha="left",
                  fontsize=11, fontweight="bold", color=INK)
    fig_.text(0.012, 0.012,
              "Next-bar log range, 15 cryptocurrencies, 4-hour bars, four test blocks (22 Apr 2025 - 14 Oct 2026, 48,600 forecasts). A run is one training seed (seeds 0-4). "
              "Price: high, low, close and range of the bar; market: volume, trades, taker imbalance;\ncalendar: time of day and day of week; technical: RSI, MACD, NATR, "
              "Bollinger %b; social and sentiment: vendor data. Every network has 20,000 parameters. H-QLSTM + LightGBM: average of the two forecasts on the same inputs.\n"
              "Change: loss of the average of the runs against the same model on the first row (positive = worse); * p < 0.05, cross-coin DM test (MSE only). "
              "Bold: lowest mean of the column.", fontsize=7.2, color=MUTED, ha="left", linespacing=1.4)
    fig_.subplots_adjust(left=0.012, right=0.988, top=0.925, bottom=0.105)
    fig_.savefig(FIG, dpi=220, facecolor="white")
    plt.close(fig_)
    print("written", FIG)


if __name__ == "__main__":
    {"check": check, "lgbm": lgbm, "fig": fig}.get((sys.argv[1:] or [""])[0], main)()
