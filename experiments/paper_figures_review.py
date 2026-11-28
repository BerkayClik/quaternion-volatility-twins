"""
Figures added after the review, from saved CSVs only.

    .venv/bin/python experiments/paper_figures_review.py   # -> docs/paper/manuscript/figures/fig_algebra.pdf

(a) quaternion/real twin ratio against the average number of training bars per model
    (log scale), with the single-seed range; (b) twin ratio against the parameter budget.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# same style as the other paper figures (experiments/paper_figures_q1.py)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, ORANGE = "#0072B2", "#E69F00"

T = Path("docs/paper/manuscript/tables")
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24}


def train_bars():
    """Average training bars per fold for one coin (single) and for all coins (pooled):
    total bars minus the validation block minus on average 2.5 test blocks."""
    d = pd.read_csv(T / "datasets.csv")
    out = {}
    for f, k in BARS_PER_DAY.items():
        per = (d[f"{f}_bars"] - k * (180 + 2.5 * 135)).clip(lower=0)
        out[f"single {f}"] = per.mean()
        out[f"pooled {f}"] = per.sum()
    return out


def main():
    tw = pd.read_csv(T / "final_twins.csv")
    tw = tw[~tw.pair.str.contains("group attn|concat")]
    bars = train_bars()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    style = {"QLSTM / LSTM": ("o", BLUE, "--"), "QTransformer / Transformer": ("s", ORANGE, "--"),
             "H-QLSTM (meta) / H-LSTM (meta)": ("o", BLUE, "-"),
             "H-QTransformer (meta) / H-Transformer (meta)": ("s", ORANGE, "-")}
    for pair, g in tw.groupby("pair"):
        g = g.assign(x=g.setting.map(bars)).sort_values("x")
        m, c, ls = style[pair]
        ax[0].plot(g.x, g.ratio, marker=m, ms=4, color=c, ls=ls, lw=1, label=pair.replace(" (meta)", ""))
        ax[0].vlines(g.x, g.seed_min, g.seed_max, color=c, lw=0.8, alpha=0.6)   # single-seed range
    ax[0].axhline(1, color="k", lw=0.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("average training bars per model (log scale)")
    ax[0].set_ylabel("MSE ratio, quaternion / real")
    ax[0].set_title("(a) Data size")
    b = pd.read_csv(T / "final_budget.csv")
    ctrl = b[b.r_hidden.notna()]          # 2k bracket: real twin widened to the quaternion's H
    b = b[b.r_hidden.isna()]
    for pair, g in b.groupby("pair"):
        g = g.sort_values("budget")
        m, c, ls = style[pair]
        ax[1].plot(g.budget, g.ratio, marker=m, ms=4, color=c, ls=ls, lw=1, label=pair.replace(" (meta)", ""))
    for _, r in ctrl.iterrows():
        m, c, _ = style[r.pair]
        ax[1].plot(r.budget, r.ratio, marker=m, ms=5, ls="none", mfc="white", mec=c, mew=1,
                   label=f"{r.pair.replace(' (meta)', '')}, R: $H={int(r.r_hidden)}$")
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set_xscale("log")
    ax[1].set_xticks([2000, 5000, 20000, 80000], ["2k", "5k", "20k", "80k"])
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_xlabel("parameter budget (pooled, 4-hourly)")
    ax[1].set_title("(b) Parameter budget")
    h, lab = ax[1].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig("docs/paper/manuscript/figures/fig_algebra.pdf")


if __name__ == "__main__":
    main()
