"""
Figures of paper v2 that are computed from the data (not from forecasts).

    .venv/bin/python -W ignore experiments/paper_v2_figures.py [data|twins|scope|all]

data   figures/fig_data.pdf: the target and the evaluation design on the full BTC 4-hourly sample
       (validation and test blocks of the four walk-forward folds), and the four feature groups,
       one quaternion each, over three weeks of the last test block.
twins  figures/fig_twins.pdf: every quaternion-against-real test of the paper, from
       tables/v2_twins_all.csv (written by paper_v2_round2.py twins): MSE ratio with its 90% interval.
scope  figures/fig_scope.pdf: MSE relative to HAR of the hierarchical recurrent twins against the parameter
       budget and the learning rate (tables/v2_scope.csv, written by paper_v2_round2.py scope).

Colours: Okabe-Ito blue, orange, green, vermillion in this fixed order for the components
(r, i, j, k); the order passes the colour-vision checks of the palette validator (worst adjacent
pair Delta E 11.0 under deuteranopia). Orange is below 3:1 against white, so every panel names its
series in a legend.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.paper.data import GROUPS, make_dataset, walk_forward_folds  # noqa: E402

OUT_F = Path("docs/paper/manuscript_v2/figures")
COMPONENT_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]          # r, i, j, k
FOLDS = dict(n_folds=4, test_days=135, val_days=180)                      # configs/paper/v2/*.yaml
PANELS = [                                                                # group, title, names of (r, i, j, k)
    ("price", "Price (Binance)", ["0 (real part)", "high", "low", "close"]),
    ("market", "Market (Binance)", ["volume", "trades", "taker imbalance", r"$\Delta$ volume"]),
    ("social", "Social (LunarCrush)", ["contributors active", "contributors created", "posts active", "posts created"]),
    ("sentiment", "Sentiment (LunarCrush)", ["sentiment", "Galaxy Score", "social dominance", "interactions"]),
]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def fig_data(coin: str = "btc", freq: str = "4h", zoom_days: int = 21):
    d = make_dataset(coin, freq, root="data")
    f, folds = d.frame, walk_forward_folds(d.frame.index, **FOLDS)
    last = folds[-1]
    train = f.iloc[last["train"]]

    fig = plt.figure(figsize=(7.2, 3.95))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.92, wspace=0.16)

    # (a) the target over the whole sample, with the blocks of the four folds
    ax = fig.add_subplot(grid[0, :])
    ax.plot(f.index, f["log_range"], color="0.62", lw=0.35, rasterized=True)
    ax.plot(f.index, f["log_range"].rolling(42).mean(), color="0.15", lw=0.9)       # one week of 4-hour bars
    for k, fold in enumerate(folds):
        ax.axvspan(fold["test_start"], fold["test_end"], color=COMPONENT_COLOURS[0], alpha=0.16, lw=0)
        ax.axvline(fold["test_start"], color="white", lw=1.2)                 # gap between adjacent blocks
        mid = fold["test_start"] + (fold["test_end"] - fold["test_start"]) / 2
        ax.text(mid, 1.02, f"test {k + 1}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    first = folds[0]
    val_start = first["test_start"] - pd.Timedelta(days=FOLDS["val_days"])
    ax.axvspan(val_start, first["test_start"], color="0.55", alpha=0.16, lw=0)
    ax.text(val_start + (first["test_start"] - val_start) / 2, 1.02, "valid. 1",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    ax.text(f.index[0] + (val_start - f.index[0]) / 2, 1.02, "training block of fold 1 (expands by one test block per fold)",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("log range")
    ax.set_xlim(f.index[0], f.index[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"(a) {coin.upper()}, four-hour bars: log range (grey) and its one-week mean (black)", loc="left", pad=14)

    # (b)-(e) the four quaternions over three weeks around the most volatile bar of the last test block
    test = f.iloc[last["test"]]
    centre = test["log_range"].idxmax()
    half = pd.Timedelta(days=zoom_days / 2)
    lo, hi = max(centre - half, test.index[0]), min(centre + half, test.index[-1])
    z = f.loc[lo:hi]
    for n, (group, title, names) in enumerate(PANELS):
        ax = fig.add_subplot(grid[1 + n // 2, n % 2])
        for col, name, colour in zip(GROUPS[group], names, COMPONENT_COLOURS):
            sd = train[col].std()
            series = (z[col] - train[col].mean()) / (sd if sd > 0 else 1.0)       # train-block z-score, as the models see it
            ax.plot(z.index, series, color=colour, lw=0.9, label=name)
        ax.axhline(0, color="0.8", lw=0.4, zorder=0)
        ax.set_title(f"({'bcde'[n]}) {title}", loc="left", pad=21)
        ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.99), frameon=False, handlelength=1.6,
                  columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.25)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        if n % 2 == 0:
            ax.set_ylabel("z-score")
    fig.savefig(OUT_F / "fig_data.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"fig_data: {coin} {freq}, zoom {lo:%Y-%m-%d} to {hi:%Y-%m-%d}, most volatile test bar {centre:%Y-%m-%d %H:%M}")


OUT_T = Path("docs/paper/manuscript_v2/tables")
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "0.35"      # quaternion, real (as in fig_depth), not significant
# Rows of the figure: (heading, [(group in v2_twins_all.csv, pair, label or None = pair)]). The figure shows the
# primary comparisons; the other rows of the file (unmatched Transformer twins, one-layer controls that the
# proposed pair repeats, secondary convolutional settings) enter the Holm adjustment and are summarised in the text.
M, MT, C, MTC = "Pooled, 20,000 parameters", "Matched Transformer twins", "With the calendar term", "Matched Transformer twins, calendar"
OTHER, FUS, SRC = "Proposed pair, other settings", "Other fusion operators (2 layers, calendar)", "Fewer sources (concatenation, 2 layers, calendar)"
B1, LR1, RND = "Other budgets (1 layer)", "Other learning rates (1 layer)", "Random feature groups (1 layer)"
HL = "Hierarchical LSTM"
TWIN_FIGURE = [
    ("Main grid: pooled, 20,000 parameters", [
        (M, "LSTM, 1 layer", None), (M, "LSTM, 2 layers", None), (M, "LSTM, 3 layers", None),
        (M, f"{HL}, 1 layer", None), (M, f"{HL}, 2 layers", None),
        (MT, "Transformer, 2 layers, and size", "Transformer, 2 layers"), (MT, "Hierarchical Transformer, 1 layer", None),
        (M, "CNN, 2 conv.", None), (M, "CNN, 4 conv.", None), (M, "CNN, 6 conv.", None),
        (M, "Hierarchical CNN, 2 conv.", None), (M, "Hierarchical CNN, 4 conv.", None)]),
    ("With the calendar term", [
        (C, "LSTM, 1 layer", None), (C, f"{HL}, 1 layer", None), (C, f"{HL}, 2 layers", "H-LSTM, 2 layers (proposed pair)"),
        (MTC, "Transformer, 2 layers", None), (MTC, "Hierarchical Transformer, 1 layer", None)]),
    ("Proposed pair in other settings", [
        (OTHER, "2,000 parameters", None), (OTHER, "5,000 parameters", None), (OTHER, "80,000 parameters", None),
        (OTHER, "Learning rate 3e-3", None), (OTHER, "Hourly bars", None), (OTHER, "Daily bars", None)]),
    ("Proposed design with another fusion operator", [
        (FUS, "Concatenation", None), (FUS, "Dense", None), (FUS, "Hamilton + tanh", None), (FUS, "Dense + tanh", None),
        (FUS, "Hamilton, projected real twin", None)]),
    ("Proposed design with fewer sources (concatenation)", [
        (SRC, "Price, market", None), (SRC, "Price, market, social", None), (SRC, "Price, market, sentiment", None)]),
    ("80,000 parameters", [("80,000 parameters", "LSTM, 3 layers", None), ("80,000 parameters", f"{HL}, 2 layers", None)]),
    ("One model per coin", [("One model per coin", f"{fam}, {d}", None) for fam in ("LSTM", HL) for d in ("1 layer", "2 layers")]),
    ("Daily bars", [("Daily bars", f"{fam}, {d}", None) for fam in ("LSTM", HL) for d in ("1 layer", "2 layers")]),
    ("One-layer hierarchical pair at other budgets", [(B1, f"{HL}, 1 layer, {n} parameters", f"H-LSTM, 1 layer, {n}") for n in ("2,000", "5,000", "80,000")]),
    ("One-layer pairs at other learning rates", [(LR1, f"{fam}, 1 layer, learning rate {r}", None) for fam in ("LSTM", HL) for r in ("3e-4", "3e-3")]),
    ("Random partition of the 16 features into groups", [(RND, f"{HL}, partition {i}", f"H-LSTM, 1 layer, partition {i}") for i in (1, 2, 3)]),
]


def fig_twins(xlim: tuple[float, float] = (0.93, 1.05)):
    """Forest plot of the primary twin tests: ratio of the MSE of the quaternion model to that of its real twin
    with the 90% interval; the band is the 1% equivalence margin; filled markers are significant after the Holm
    adjustment over all rows of v2_twins_all.csv."""
    t = pd.read_csv(OUT_T / "v2_twins_all.csv")
    t["group"] = t.group.replace({"Hourly bars": OTHER})
    daily_proposed = (t.group == "Daily bars") & t.pair.str.contains("calendar")
    t.loc[daily_proposed, "group"] = OTHER
    t.loc[daily_proposed, "pair"] = "Daily bars"
    t.loc[(t.bars == "1h"), "pair"] = "Hourly bars"
    short = lambda v: v.replace("Hierarchical ", "H-").replace(", learning rate", ", lr")
    lines, shown = [], 0                        # (kind, text, row)
    for title, rows in TWIN_FIGURE:
        found = []
        for group, pair, label in rows:
            hit = t[(t.group == group) & (t.pair == pair)]
            if len(hit) == 1:
                found.append(("row", short(label or pair), hit.iloc[0]))
            else:
                print(f"not in v2_twins_all.csv: {group} / {pair}")
        if found:
            lines.append(("head", title, None))
            lines += found
            shown += len(found)
    heads = [i for i, line in enumerate(lines) if line[0] == "head"]
    half = min(heads, key=lambda i: abs(i - len(lines) / 2))      # the second column starts with a heading
    cols = [lines[:half], lines[half:]]
    n = max(len(c) for c in cols)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.089 * n + 0.55), sharex=True)
    for ax, col in zip(axes, cols):
        ax.axvspan(0.99, 1.01, color="0.90", lw=0, zorder=0)
        ax.axvline(1.0, color="0.45", lw=0.6, zorder=1)
        for i, (kind, text, r) in enumerate(col):
            y = n - 1 - i
            if kind == "head":
                ax.text(-0.60, y, text, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.6,
                        style="italic", color="0.1")
                continue
            sig = r.p_holm < 0.05
            colour = (ORANGE if r.pooled > 1 else BLUE) if sig else GREY
            lo, hi = max(r.ci_lo, xlim[0]), min(r.ci_hi, xlim[1])
            ax.plot([lo, hi], [y, y], color=colour, lw=1.0, solid_capstyle="butt", zorder=2)
            for edge, clipped, marker in ((xlim[0], r.ci_lo < xlim[0], "<"), (xlim[1], r.ci_hi > xlim[1], ">")):
                if clipped:                      # interval runs past the axis
                    ax.plot([edge], [y], marker=marker, ms=2.6, color=colour, clip_on=False, zorder=3)
            ax.plot([min(max(r.pooled, xlim[0]), xlim[1])], [y], marker="o", ms=3.4, mfc=colour if sig else "white", mec=colour,
                    mew=0.8, zorder=4)
            ax.text(-0.58, y, text, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.4, color="0.1")
            ax.text(1.03, y, f"{r.pooled:.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.2,
                    color="0.1")
            ax.text(1.34, y, f"{int(r.better)}/{int(r.n)}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=6.2, color="0.1")
        ax.text(1.03, n - 0.1, "ratio", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6.2, color="0.3")
        ax.text(1.34, n - 0.1, "coins", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.2, color="0.3")
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks([0.94, 0.96, 0.98, 1.0, 1.02, 1.04])
        ax.tick_params(axis="x", length=2, width=0.5)
        ax.set_xlabel("MSE of quaternion model / MSE of real twin", fontsize=7)
    handles = [plt.Line2D([], [], color=BLUE, marker="o", ms=3.4, lw=1.0, label="quaternion significantly better"),
               plt.Line2D([], [], color=ORANGE, marker="o", ms=3.4, lw=1.0, label="real twin significantly better"),
               plt.Line2D([], [], color=GREY, marker="o", ms=3.4, mfc="white", lw=1.0, label="not significant"),
               plt.Rectangle((0, 0), 1, 1, color="0.90", label="equivalence margin (0.99 to 1.01)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(left=0.153, right=0.910, wspace=1.02, top=1 - 0.30 / fig.get_figheight(), bottom=0.38 / fig.get_figheight())
    fig.savefig(OUT_F / "fig_twins.pdf")
    plt.close(fig)
    print("fig_twins.pdf:", shown, "of", len(t), "tests shown;", int((t.p_holm < 0.05).sum()), "significant in all")



def fig_scope():
    """MSE relative to HAR of the hierarchical recurrent twins against the parameter budget and the learning
    rate (tables/v2_scope.csv, written by paper_v2_round2.py scope): the one-layer pair of the first stage,
    without the calendar term, and the proposed pair."""
    t = pd.read_csv(OUT_T / "v2_scope.csv")
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.0))          # one column of the page
    styles = (("real", ORANGE, "o"), ("quaternion", BLUE, "s"))
    panels = [("budget", "H-LSTM, 1 layer", "One layer: budget", "trainable parameters"),
              ("learning rate", "H-LSTM, 1 layer", "One layer: learning rate", "learning rate"),
              ("budget", "Proposed pair", "Proposed pair: budget", "trainable parameters"),
              ("learning rate", "Proposed pair", "Proposed pair: learning rate", "learning rate")]
    for ax, (panel, pair, title, label) in zip(axes.ravel(), panels):
        g = t[(t.pair == pair) & (t.panel == panel)]
        for algebra, colour, marker in styles:
            h = g[g.algebra == algebra].sort_values("x")
            ax.plot(h.x, h.mse_har, color=colour, marker=marker, ms=4, lw=1.2)
        xs = sorted(g.x.unique())
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(v / 1000)}k" for v in xs] if panel == "budget" else [f"{v:g}" for v in xs])
        ax.minorticks_off()
        ax.margins(x=0.14)
        lo, hi = g.mse_har.min(), g.mse_har.max()
        mid, half = (lo + hi) / 2, max((hi - lo) / 2 * 1.25, 0.012)          # at least 2.4 points of MSE / HAR
        ax.set_ylim(mid - half, mid + half)
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.grid(axis="y", lw=0.3, color="0.85")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE / HAR")
    handles = [plt.Line2D([], [], color=ORANGE, marker="o", ms=4, lw=1.2, label="real"),
               plt.Line2D([], [], color=BLUE, marker="s", ms=4, lw=1.2, label="quaternion")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.8, h_pad=0.7)
    fig.savefig(OUT_F / "fig_scope.pdf")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("data", "all"):
        fig_data()
    if which in ("twins", "all"):
        fig_twins()
    if which in ("scope", "all"):
        fig_scope()
