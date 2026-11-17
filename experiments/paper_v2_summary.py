"""
Shared loaders and statistics of the v2 analysis (load, run, pooled_ratio, twin_table), and a one-page
summary of the first v2 runs (pooled 4-hour bars, 15 coins, seeds 0-2).

    .venv/bin/python -W ignore experiments/paper_v2_summary.py [--out docs/paper/review/advisor_2026-11-16]

Writes v2_results_summary.tex (a standalone document; compile with tectonic) and the two tables as CSV.
Every number is computed here from the saved forecasts:
  results/final_fa, final_local   runs of the first stage (HAR family, LightGBM, flat networks)
  results/v2, results/v2cal       v2 depth / CNN grid and calendar grid
Statistics: MSE of the seed-averaged forecast relative to HAR, mean over coins; twin ratio = mean over
coins of MSE(quaternion) / MSE(real); cross-coin Diebold-Mariano test; Holm adjustment over the rows of
the twin table. The tables of the paper are written by paper_v2_round2.py and paper_v2_tables.py.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from src.paper.evaluate import dm_cross, dm_table, holm, load_run, seed_mean  # noqa: E402
from paper_round8_checks import _ratio_se  # noqa: E402  (ratio of coin-averaged losses, delta-method NW se)

MARGIN = 0.01                 # equivalence margin of paper v1: 90% interval of the ratio inside 1 +- 1%

HAR = ("P", "har")
N_CELLS = 15 * 3 * 4          # coins x seeds x folds of a complete pooled model


def run(pattern: str) -> pd.DataFrame:
    dirs = glob.glob(pattern)
    if len(dirs) != 1:
        raise SystemExit(f"expected one run for {pattern}, found {dirs}")
    d = load_run(dirs[0])
    return d[(d.task == "range") & (d.seed <= 2)]


def load() -> pd.DataFrame:
    single = run("results/final_fa/out/final_single_4h_*")
    depth = run("results/v2/out/v2_depth_pool_4h_*").query("model != 'har'")
    # v1 pooled run: only the models that the v2 grid did not train again (the Transformers);
    # the one-layer recurrent models come from the v2 grid, like the deeper ones
    v1_pool = run("results/final_fa/out/final_pool_4h_*").query("groupset == 'PMSE'")
    v1_pool = v1_pool[~v1_pool.model.isin(depth.model.unique())]
    parts = [single[(single.model == "har") & (single.groupset == "P")], v1_pool,
             run("results/final_local/final_harx_4h_*"), run("results/final_local/final_har_season_4h_*"),
             run("results/final_local/final_lgbm_4h_*"),
             # seeds 1-2 of the pooled LightGBM-S, so that the benchmark is a three-seed average like the networks
             run("results/final_local/final_lgbm_seeds_4h_*"), run("results/v2local/v2_lgbm_seeds_4h_*"),
             run("results/final_fc/out/final_calendar_pool_4h_*").query("groupset == 'PMC'"),
             depth, run("results/v2cal/out/v2_cal_pool_4h_*")]
    df = pd.concat(parts, ignore_index=True)
    dup = df.duplicated(["dataset", "groupset", "model", "seed", "fold", "time"]).sum()
    assert dup == 0, f"{dup} duplicated forecasts: two runs supply the same model"
    pooled = df[df.model.str.endswith("-pool") & ~df.model.str.startswith(("har", "lgbm"))]
    n = pooled.groupby(["groupset", "model"]).apply(lambda g: len(g[["dataset", "seed", "fold"]].drop_duplicates()),
                                                     include_groups=False)
    incomplete = set(n[n < N_CELLS].index)
    if incomplete:
        print("incomplete, left out:", sorted(incomplete))
    keep = [(g, m) not in incomplete for g, m in zip(df.groupset, df.model)]
    return df[keep]


# (label, inputs, calendar, (groupset, model)); None = group heading
MAIN_ROWS = [
    ("Benchmarks", None),
    ("HAR", "past log range", "no", HAR),
    ("HAR-X", "16 features", "no", ("PMSE", "harx")),
    ("LightGBM, pooled", "16 features", "no", ("PMSE", "lgbm-pool")),
    ("HAR-S", "past log range", "dummies", ("PMSE", "hars")),
    ("HAR-XS", "16 features", "dummies", ("PMSE", "harxs")),
    ("LightGBM-S, pooled", "16 features", "dummies", ("PMSE", "lgbms-pool")),
    ("LightGBM-S, pooled", "price + market", "dummies", ("PM", "lgbms-pool")),
    ("Flat networks (one encoder for all features)", None),
    ("LSTM", "16 features", "no", ("PMSE", "lstm-pool")),
    ("Transformer", "16 features", "no", ("PMSE", "transformer-pool")),
    ("LSTM", "16 features", "added term", ("PMSEC", "lstm+cal-pool")),
    ("Transformer", "16 features", "added term", ("PMSEC", "transformer+cal-pool")),
    ("Proposed: one quaternion encoder per source, meta-quaternion fusion, 2 layers", None),
    ("H-QLSTM (proposed)", "16 features", "no", ("PMSE", "hq_meta-L2-pool")),
    ("H-LSTM (real twin)", "16 features", "no", ("PMSE", "hl_meta-L2-pool")),
    ("H-QLSTM (proposed)", "16 features", "added term", ("PMSEC", "hq_meta-L2+cal-pool")),
    ("H-LSTM (real twin)", "16 features", "added term", ("PMSEC", "hl_meta-L2+cal-pool")),
    ("Best model of paper v1, for reference", None),
    ("H-LSTM, group attention, 1 layer", "price + market", "encoder group", ("PMC", "hl_gattn-pool")),
]

# (family, depth / setting, quaternion model, real model, groupset)
TWIN_ROWS = [
    ("LSTM", "1 layer", "qlstm", "lstm", "PMSE"), ("LSTM", "2 layers", "qlstm-L2", "lstm-L2", "PMSE"),
    ("LSTM", "3 layers", "qlstm-L3", "lstm-L3", "PMSE"),
    ("Hierarchical LSTM", "1 layer", "hq_meta", "hl_meta", "PMSE"),
    ("Hierarchical LSTM", "2 layers", "hq_meta-L2", "hl_meta-L2", "PMSE"),
    ("Transformer", "2 layers", "qtransformer", "transformer", "PMSE"),
    ("Hierarchical Transformer", "1 layer", "hqt_meta", "ht_meta", "PMSE"),
    ("CNN", "2 conv.", "qcnn-L2", "cnn-L2", "PMSE"), ("CNN", "4 conv.", "qcnn", "cnn", "PMSE"),
    ("CNN", "6 conv.", "qcnn-L6", "cnn-L6", "PMSE"),
    ("Hierarchical CNN", "2 conv.", "hqc_meta-L2", "hc_meta-L2", "PMSE"),
    ("Hierarchical CNN", "4 conv.", "hqc_meta", "hc_meta", "PMSE"),
    ("LSTM + calendar", "1 layer", "qlstm+cal", "lstm+cal", "PMSEC"),
    ("Transformer + calendar", "2 layers", "qtransformer+cal", "transformer+cal", "PMSEC"),
    ("Hierarchical LSTM + calendar", "1 layer", "hq_meta+cal", "hl_meta+cal", "PMSEC"),
    ("Hierarchical LSTM + calendar", "2 layers", "hq_meta-L2+cal", "hl_meta-L2+cal", "PMSEC"),
    ("Hierarchical Transformer + calendar", "1 layer", "hqt_meta+cal", "ht_meta+cal", "PMSEC"),
]


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    d = dm_table(df, {"range": HAR})
    s = d.groupby(["groupset", "model"]).agg(mse=("rel_loss", "mean"), better=("rel_loss", lambda v: int((v < 1).sum())),
                                             n=("rel_loss", "size"))
    rows = []
    for row in MAIN_ROWS:
        if row[1] is None:
            rows.append(dict(label=row[0], heading=True))
        elif row[3] == HAR:
            rows.append(dict(label=row[0], inputs=row[1], calendar=row[2], mse=1.0, better=None, n=15, heading=False))
        elif row[3] in s.index:
            r = s.loc[row[3]]
            rows.append(dict(label=row[0], inputs=row[1], calendar=row[2], mse=r.mse, better=int(r.better), n=int(r.n),
                             heading=False))
        else:
            rows.append(dict(label=row[0], inputs=row[1], calendar=row[2], mse=None, heading=False))
    return pd.DataFrame(rows)


def pooled_ratio(x: pd.DataFrame, a: tuple, b: tuple) -> tuple[float, float, float]:
    """Ratio of the coin-averaged MSE of a to that of b on the common dates, with its 90% interval."""
    ens = seed_mean(x).assign(se=lambda v: (v.pred - v.y) ** 2)
    loss = {k: ens[(ens.groupset == k[0]) & (ens.model == k[1])].pivot(index="time", columns="dataset", values="se")
            for k in (a, b)}
    dates = loss[a].dropna().index.intersection(loss[b].dropna().index)
    r, _, se = _ratio_se(loss[a].loc[dates].mean(axis=1), loss[b].loc[dates].mean(axis=1))
    return r, r - 1.645 * se, r + 1.645 * se


def twin_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, depth, q, r, gs in TWIN_ROWS:
        a, b = (gs, q + "-pool"), (gs, r + "-pool")
        x = df[[(g, m) in (a, b) for g, m in zip(df.groupset, df.model)]]
        if x.model.nunique() < 2:
            rows.append(dict(family=family, depth=depth, ratio=None))
            continue
        t = dm_table(x, {"range": b})
        c = dm_cross(x, a, b)
        _, lo, hi = pooled_ratio(x, a, b)
        rows.append(dict(family=family, depth=depth, ratio=t.rel_loss.mean(), q_better=int((t.rel_loss < 1).sum()),
                         n=len(t), dm=c["dm_stat"], p=c["p"], ci_lo=lo, ci_hi=hi,
                         equivalent=bool(lo > 1 - MARGIN and hi < 1 + MARGIN)))
    t = pd.DataFrame(rows)
    ok = t.ratio.notna()
    t.loc[ok, "p_holm"] = holm(t.loc[ok, "p"])
    return t


def tex(main: pd.DataFrame, twins: pd.DataFrame) -> str:
    out = [r"\documentclass[10pt]{article}", r"\usepackage[a4paper,margin=1.7cm]{geometry}",
           r"\usepackage{booktabs,amsmath}", r"\usepackage{newtxtext,newtxmath}", r"\pagestyle{empty}",
           r"\setlength{\parindent}{0pt}", r"\begin{document}",
           r"{\large\bfseries Paper v2: results of the new experiments (16 November 2026)}\\[3pt]",
           r"15 coins pooled in one model, 4-hour bars, next-bar log range, three seeds, four walk-forward test blocks "
           r"(540 days), about 20{,}000 parameters per network. All numbers are out of sample.\\[8pt]",
           r"\textbf{Table 1. Proposed model against benchmarks.} MSE relative to HAR (lower is better), mean over the "
           r"15 coins; `better' counts the coins on which the model beats HAR.\\[3pt]",
           r"\begin{tabular}{@{}lllrr@{}}", r"\toprule",
           r"Model & Inputs & Calendar & MSE / HAR & Better \\", r"\midrule"]
    first = True
    for _, r in main.iterrows():
        if r.heading:
            if not first:
                out.append(r"\addlinespace[3pt]")
            out.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{r.label}}}}} \\")
            first = False
        elif pd.isna(r.mse):
            out.append(rf"\quad {r.label} & {r.inputs} & {r.calendar} & running & \\")
        else:
            better = "" if pd.isna(r.better) else f"{int(r.better)}/{int(r.n)}"
            bold = r"\textbf" if "proposed" in r.label else ""
            out.append(rf"\quad {r.label} & {r.inputs} & {r.calendar} & {bold}{{{r.mse:.3f}}} & {better} \\")
    out += [r"\bottomrule", r"\end{tabular}\\[4pt]",
            r"{\footnotesize Calendar: time of day and day of week of the bar to be forecast. `dummies': calendar dummies in "
            r"the benchmark; `added term': a small network adds a function of the calendar to the forecast; `encoder "
            r"group': calendar features pass through their own encoder. The calendar runs were added after the first "
            r"test results had been seen.}\\[10pt]",
            r"\textbf{Table 2. Quaternion model against its real twin} (same architecture, depth, inputs and number of "
            r"parameters). Ratio below 1: the quaternion model is better.\\[3pt]",
            r"\begin{tabular}{@{}llrrrll@{}}", r"\toprule",
            r"Encoder family & Depth & Ratio Q / R & Q better & DM & Significant (Holm, 5\%) & Within $\pm$1\% \\",
            r"\midrule"]
    last = None
    for _, r in twins.iterrows():
        if last is not None and ("calendar" in r.family) != ("calendar" in last):
            out.append(r"\addlinespace[3pt]")
        last = r.family
        if pd.isna(r.ratio):
            out.append(rf"{r.family} & {r.depth} & running & & & & \\")
            continue
        sig = "no" if r.p_holm >= 0.05 else ("yes, real better" if r.ratio > 1 else "yes, quaternion better")
        eq = "yes" if r.equivalent else "not shown"
        out.append(rf"{r.family} & {r.depth} & {r.ratio:.3f} & {int(r.q_better)}/{int(r.n)} & {r.dm:+.2f} & {sig} & {eq} \\")
    done = twins[twins.ratio.notna()]
    n_sig_r = int(((done.p_holm < 0.05) & (done.ratio > 1)).sum())
    n_sig_q = int(((done.p_holm < 0.05) & (done.ratio < 1)).sum())
    out += [r"\bottomrule", r"\end{tabular}\\[4pt]",
            rf"{{\footnotesize DM: Diebold--Mariano test on the loss differential averaged over coins; Holm adjustment over "
            rf"the {len(done)} rows. Significant rows: {n_sig_r} in favour of the real twin, {n_sig_q} in favour of the "
            rf"quaternion model. Within $\pm$1\%: the 90\% interval of the ratio of coin-averaged losses lies inside "
            rf"0.99--1.01, so the two models are equivalent at that margin ({int(done.equivalent.sum())} of "
            rf"{len(done)} rows); `not shown' means that equivalence is not established, not that the models differ.}}",
            r"\end{document}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper/review/advisor_2026-11-16")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load()
    main, twins = main_table(data), twin_table(data)
    main.to_csv(out / "v2_summary_main.csv", index=False)
    twins.to_csv(out / "v2_summary_twins.csv", index=False)
    (out / "v2_results_summary.tex").write_text(tex(main, twins))
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(main[~main.heading].drop(columns=["heading"]).round(4).to_string(index=False))
    print()
    print(twins.round(4).to_string(index=False))
