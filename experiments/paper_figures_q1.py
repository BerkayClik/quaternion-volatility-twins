"""
Submission figures computed from saved forecasts and review CSVs.

    .venv/bin/python -W ignore experiments/paper_figures_q1.py [forest|ladder|equity|ga|all]

Outputs (docs/paper/manuscript/figures/): fig_forest.pdf, fig_ladder.pdf, fig_equity.pdf,
graphical_abstract.{pdf,png}; numbers behind them in tables/fig_{forest,ladder,equity}.csv.

forest  quaternion/real twin ratios (log-range MSE) with 90% Newey-West intervals and the
        single-seed range, for the 16-feature (PMSE) and price-only (P) twins.
ladder  MSE change of each design choice (median over model pairs, min-max whiskers).
equity  volatility-targeted 4h strategies of the economic evaluation, equal-weight over coins.
ga      graphical abstract (1328 x 531 px).
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from paper_tables_final import OUT_F, OUT_T, POOL, TWINS, _pair, setting  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
# Okabe-Ito colour-blind safe palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
FREQS = ("1h", "4h", "1d")
FNAME = {"1h": "Hourly", "4h": "4-hourly", "1d": "Daily"}
SHORT = {"qlstm/lstm": "QLSTM / LSTM", "qtransformer/transformer": "QTransf. / Transf.",
         "hq_meta/hl_meta": "H-QLSTM / H-LSTM (meta)", "hqt_meta/ht_meta": "H-QT / H-T (meta)",
         "hq_gattn/hl_gattn": "H-QLSTM / H-LSTM (attn.)", "hqt_concat/ht_concat": "H-QT / H-T (concat)"}


# ----------------------------------------------------------------------------- forest
def forest_numbers():
    """Twin ratios on the cross-coin pooled loss (the basis of the equivalence test): seed
    ensemble with 90% NW interval, and the same ratio for every single seed."""
    from paper_review_checks import equivalence
    path = OUT_T / "fig_forest.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for f in FREQS:
        df = setting(f)
        for tag, sfx in (("single", ""), ("pooled", POOL)):
            for gs in ("P", "PMSE"):
                for q, r in TWINS:
                    a, b = (gs, q + sfx), (gs, r + sfx)
                    x = df[[(g, m) in (a, b) for g, m in zip(df.groupset, df.model)]]
                    if x.model.nunique() < 2:
                        continue
                    e = equivalence(x, a, b)
                    per = [equivalence(x[x.seed == k], a, b)["ratio"] for k in sorted(x.seed.unique())]
                    rows.append(dict(freq=f, setting=tag, inputs=gs, pair=f"{q}/{r}", **e,
                                     seed_min=min(per), seed_max=max(per), n_seeds=len(per)))
        print("forest", f, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(path, index=False)
    return t


def fig_forest():
    t = forest_numbers()
    order = [("single", "P"), ("single", "PMSE"), ("pooled", "P"), ("pooled", "PMSE")]
    pairs = list(SHORT)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.1), sharey=True)
    ylabels, y = [], 0
    ypos = {}
    for st, gs in order:
        for p in pairs:
            if ((t.setting == st) & (t.inputs == gs) & (t.pair == p)).any():
                ypos[(st, gs, p)] = y
                ylabels.append(SHORT[p])
                y += 1
        y += 0.6
    for ax, f in zip(axes, FREQS):
        ax.axvspan(0.99, 1.01, color=C["grey"], alpha=0.15, lw=0)
        ax.axvline(1, color="0.45", lw=0.5, zorder=1)   # below the markers, so a marker at 1.00 stays visible
        for (st, gs, p), yy in ypos.items():
            r = t[(t.freq == f) & (t.setting == st) & (t.inputs == gs) & (t.pair == p)]
            if r.empty:
                continue
            r = r.iloc[0]
            col = C["orange"] if gs == "P" else C["blue"]
            ax.hlines(yy, r.seed_min, r.seed_max, color=col, lw=0.6, alpha=0.7, zorder=2)
            ax.errorbar(r.ratio, yy, xerr=[[r.ratio - r.ci_lo], [r.ci_hi - r.ratio]], fmt="none",
                        ecolor=col, elinewidth=1.6, capsize=0, zorder=3)
            # equivalent: filled circle; not equivalent: open diamond (shape survives low resolution and greyscale)
            if r.equivalent:
                ax.plot(r.ratio, yy, "o", ms=5.2, mec="k", mfc=col, mew=0.6, zorder=5)
            else:
                ax.plot(r.ratio, yy, "D", ms=4.6, mec=col, mfc="white", mew=1.1, zorder=5)
        ax.set_title(FNAME[f])
        ax.set_xlabel("MSE ratio, quaternion / real")
        ax.set_xlim(0.96, 1.075)
        ax.set_xticks([0.97, 1.0, 1.03, 1.06])
    # section labels and separators
    for st, gs in order:
        ys = [v for (s, g, _), v in ypos.items() if (s, g) == (st, gs)]
        axes[0].text(-0.02, min(ys) - 0.55, f"{st.capitalize()}, {'price only (P)' if gs == 'P' else 'all 16 features (PMSE)'}",
                     transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=7, fontweight="bold")
        for ax in axes:
            ax.axhline(min(ys) - 0.8, color=C["grey"], lw=0.3, ls=":")
    axes[0].set_yticks(list(ypos.values()), ylabels)
    axes[0].invert_yaxis()
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=5.2, mfc="0.5", mec="k", mew=0.6, label="equivalent (90% CI inside $\\pm$1%)"),
         plt.Line2D([], [], marker="D", ls="none", ms=4.6, mfc="white", mec="0.3", mew=1.1, label="not equivalent"),
         plt.Line2D([], [], color="0.3", lw=1.6, label="90% NW interval"),
         plt.Line2D([], [], color="0.3", lw=0.6, label="single-seed range"),
         plt.Line2D([], [], color=C["orange"], lw=4, label="price only (P)"),
         plt.Line2D([], [], color=C["blue"], lw=4, label="all 16 features (PMSE)")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.55, -0.005), fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_F / "fig_forest.pdf", bbox_inches="tight")
    fig.savefig("/tmp/fig_forest.png", dpi=200, bbox_inches="tight")


# ----------------------------------------------------------------------------- ladder
HIER = [("hl_meta", "lstm"), ("ht_meta", "transformer"), ("hq_meta", "qlstm"), ("hqt_meta", "qtransformer")]


def ladder_numbers():
    rows = []
    add = lambda eff, f, comp, ratio: rows.append(dict(effect=eff, freq=f, comparison=comp, ratio=ratio))
    for _, r in pd.read_csv(OUT_T / "review_pooling_seedmatched.csv").iterrows():
        add("Pooling across coins", r.freq, f"{r.model} pooled/single (seeds 0-2)", r.pooled3 / r.single)
    cal = pd.read_csv(OUT_T / "review_calendar.csv")
    for _, r in cal[cal.comp.str.contains(" PMC / PM$")].iterrows():
        add("Calendar inputs", r.freq, r.comp, r.ratio)
    alt = pd.read_csv(OUT_T / "review_altdata.csv")
    for _, r in alt[alt.model.str.endswith("PM/P") & alt.freq.isin(["1h", "4h"])].iterrows():
        add("Market data (intraday)", r.freq, r.model, r.ratio)
    for _, r in alt[~alt.model.str.endswith("PM/P")].iterrows():
        add("Social + sentiment data", r.freq, r.model + " PMSE/PM", r.ratio)
    for f in FREQS:
        df = setting(f)
        for tag, sfx in (("single", ""), ("pooled", POOL)):
            for h, fl in HIER:
                a, b = ("PMSE", h + sfx), ("PMSE", fl + sfx)
                if ((df.groupset == "PMSE") & (df.model == a[1])).any() and ((df.groupset == "PMSE") & (df.model == b[1])).any():
                    add(f"Hierarchy ({'single coin' if tag == 'single' else 'pooled'})", f, f"{a[1]}/{b[1]}",
                        _pair(df, a, b)["ratio"])
        print("ladder", f, flush=True)
    tw = pd.read_csv(OUT_T / "final_twins.csv")
    for _, r in tw.iterrows():
        add("Quaternion algebra", r.setting.split()[1], f"{r.setting}: {r.pair}", r.ratio)
    fu = pd.read_csv(OUT_T / "final_fusion.csv")
    fu = fu[fu.block.isin(["linear", "nonlinear"]) & fu.a.isin(["phm", "dense", "phmn", "densen"])]
    for _, r in fu.iterrows():
        add("Fusion operator", "4h", f"{r.enc} {r.a}/{r.b} {r.gs}", r.ratio)
    t = pd.DataFrame(rows)
    t["change_pct"] = 100 * (t.ratio - 1)
    t.to_csv(OUT_T / "fig_ladder.csv", index=False)
    return t


def fig_ladder():
    t = ladder_numbers()
    s = t.groupby("effect").change_pct.agg(["median", "min", "max", "size"])
    s = s.reindex(s["median"].abs().sort_values(ascending=False).index)
    s.to_csv(OUT_T / "fig_ladder_summary.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(s))
    cols = [C["blue"] if m < 0 else C["red"] for m in s["median"]]
    ax.barh(y, s["median"], color=cols, alpha=0.85, height=0.62)
    ax.hlines(y, s["min"], s["max"], color="k", lw=0.8)
    for yy, (_, r) in zip(y, s.iterrows()):
        x = t[t.effect == r.name].change_pct.to_numpy()
        ax.plot(x, yy + np.random.default_rng(0).uniform(-0.18, 0.18, len(x)), ".", color="k", ms=1.6, alpha=0.45)
    ax.set_yticks(y, [f"{n} ({int(k)})" for n, k in zip(s.index, s["size"])])
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("change in MSE (%), median and range over model pairs")
    ax.text(0.02, 1.0, "lower error $\\leftarrow$", transform=ax.transAxes, fontsize=6.5, color=C["blue"], va="bottom")
    ax.text(0.98, 1.0, "$\\rightarrow$ higher error", transform=ax.transAxes, fontsize=6.5, color=C["red"],
            va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(OUT_F / "fig_ladder.pdf", bbox_inches="tight")
    fig.savefig("/tmp/fig_ladder.png", dpi=220, bbox_inches="tight")
    print(s.round(2).to_string())


# ----------------------------------------------------------------------------- equity
EQ_MODELS = [(("P", "har"), "HAR", C["grey"]), (("PMSE", "hars"), "HAR-S", C["sky"]),
             (("PM", "hl_gattn-pool"), "H-LSTM (group attn., pooled)", C["blue"]),
             (("PM", "combo:hl_gattn-pool"), "50/50 HAR + H-LSTM", C["green"])]


def equity_series():
    """Per-coin strategy returns exactly as in paper_review_checks.main_econ (band 0):
    w = min(2, target / sigma_hat), simple returns, 5 bp per unit turnover."""
    from paper_tables_final import run_dir
    from src.paper.economic import BARS_PER_YEAR, load_preds, variance_forecasts
    n = BARS_PER_YEAR["4h"]
    tgt = 0.40 / np.sqrt(n)
    want = [("P", "har"), ("PMSE", "hars"), ("PM", "hl_gattn-pool")]
    dirs = [run_dir(x) for x in ("final_single_4h", "final_pool_4h", "final_har_season_4h")]
    keep = lambda d: d[[(g, m) in want for g, m in zip(d.groupset, d.model)]]
    test = pd.concat([keep(load_preds(d)) for d in dirs if d])
    val = pd.concat([keep(load_preds(d, split="val")) for d in dirs if d])
    V = variance_forecasts(test, val)
    har = V[(V.groupset == "P") & (V.model == "har")].set_index(["dataset", "time"])
    g = V[(V.groupset == "PM") & (V.model == "hl_gattn-pool")].set_index(["dataset", "time"])
    idx = g.index.intersection(har.index)
    c = g.loc[idx].copy()
    c["var"] = 0.5 * (g.loc[idx, "var"] + har.loc[idx, "var"])
    c["model"] = "combo:hl_gattn-pool"
    V = pd.concat([V, c.reset_index()], ignore_index=True)
    out = []
    for (ds, gs, m), h in V.groupby(["dataset", "groupset", "model"]):
        h = h.sort_values("time")
        w = np.minimum(2.0, tgt / np.sqrt(h["var"].to_numpy()))
        simple = np.expm1(h.r.to_numpy())
        turn = np.abs(np.diff(w, prepend=w[0]))
        out.append(pd.DataFrame({"dataset": ds, "groupset": gs, "model": m, "time": h.time.to_numpy(),
                                 "rp": w * simple - 0.0005 * turn, "turn": turn, "bh": simple}))
    return pd.concat(out, ignore_index=True), n


def fig_equity():
    S, n = equity_series()
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    stats = []
    bh = S[(S.groupset == "P") & (S.model == "har")].pivot(index="time", columns="dataset", values="bh").mean(axis=1)
    curves = [("Buy and hold", C["black"], bh, None)]
    for (gs, m), lab, col in EQ_MODELS:
        x = S[(S.groupset == gs) & (S.model == m)]
        port = x.pivot(index="time", columns="dataset", values="rp").mean(axis=1)
        curves.append((lab, col, port, x))
    for lab, col, port, x in curves:
        eq = np.cumprod(1 + port.to_numpy())
        ls = ":" if lab == "Buy and hold" else "-"
        ax[0].plot(port.index, eq, color=col, lw=0.9, ls=ls, label=lab)
        lr = np.log1p(port.to_numpy())
        dd = np.max(np.maximum.accumulate(np.cumsum(lr)) - np.cumsum(lr))
        row = dict(strategy=lab, ann_return=np.mean(port) * n, ann_vol=np.std(port) * np.sqrt(n),
                   sharpe=np.mean(port) / np.std(port) * np.sqrt(n), max_drawdown=1 - np.exp(-dd),
                   final_wealth=eq[-1])
        if x is not None:
            rv = x.pivot(index="time", columns="dataset", values="rp").rolling(180).std() * np.sqrt(n)
            mean_rv = rv.mean(axis=1)
            ax[1].plot(mean_rv.index, mean_rv, color=col, lw=0.8, label=lab)
            per_coin = x.groupby("dataset").rp.std() * np.sqrt(n)
            row.update(coin_vol_err=float(np.mean(np.abs(per_coin - 0.40))),
                       rolling_track_rmse=float(np.sqrt(np.nanmean((rv - 0.40) ** 2))),
                       turnover=float(x.groupby("dataset").turn.mean().mean() * n))
        stats.append(row)
    ax[1].axhline(0.40, color="k", lw=0.6, ls="--")
    ax[0].axhline(1, color="k", lw=0.4)
    ax[0].set_ylabel("wealth (equal-weight)")
    ax[1].set_ylabel("30-day realised vol.")
    ax[1].text(ax[1].get_xlim()[0], 0.405, " target 40%", fontsize=6.5, va="bottom")
    # legend below both panels, so it covers no line
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.3, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    ax[0].set_title("(a) Volatility-targeted strategies, 4-hourly, after 5 bp costs", loc="left")
    ax[1].set_title("(b) Mean over coins of the rolling 30-day volatility", loc="left")
    ax[1].xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    ax[1].tick_params(axis="x", labelsize=6.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT_F / "fig_equity.pdf", bbox_inches="tight")
    fig.savefig("/tmp/fig_equity.png", dpi=220, bbox_inches="tight")
    t = pd.DataFrame(stats)
    t.to_csv(OUT_T / "fig_equity.csv", index=False)
    print(t.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- graphical abstract
def _box(ax, x, y, w, h, fc, ec, text=None, fs=20, color="k", weight="normal", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.5))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight)


def _arrow(ax, a, b, col="#444444"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, lw=2, color=col))


def fig_ga():
    W, H = 13.28, 5.31
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    hdr = dict(fontsize=23, weight="bold", ha="center", va="center")
    # ---- left: data -> four quaternions
    ax.text(2.05, 4.85, "Data: 15 coins", **hdr)
    groups = [("Price", "Binance", C["blue"]), ("Market", "Binance", C["sky"]),
              ("Social", "LunarCrush", C["orange"]), ("Sentiment", "LunarCrush", C["yellow"])]
    for i, (g, src, col) in enumerate(groups):
        y0 = 3.65 - i * 0.98
        _box(ax, 0.35, y0, 2.3, 0.78, col + "40", col)
        ax.text(1.5, y0 + 0.50, g, ha="center", va="center", fontsize=19)
        ax.text(1.5, y0 + 0.20, src, ha="center", va="center", fontsize=13, color="#444444")
        for k in range(4):
            ax.add_patch(plt.Circle((2.95 + k * 0.24, y0 + 0.39), 0.09, color=col))
    ax.text(3.32, 0.25, "4 features = 1 quaternion", ha="center", fontsize=16, style="italic")
    _arrow(ax, (3.95, 2.45), (4.55, 2.45))
    # ---- middle: twin design
    ax.text(6.55, 4.85, "Twin design", **hdr)
    _box(ax, 4.7, 3.25, 1.75, 1.05, "#dbe9f6", C["blue"], "Quaternion\nencoder", fs=18)
    _box(ax, 6.65, 3.25, 1.75, 1.05, "#f2f2f2", C["grey"], "Real twin\n(same size)", fs=18)
    ax.text(6.55, 2.75, "×", fontsize=26, ha="center", va="center", weight="bold")
    _box(ax, 4.7, 1.25, 1.75, 1.05, "#fff2d9", C["orange"], "Flat\n16 inputs", fs=18)
    _box(ax, 6.65, 1.25, 1.75, 1.05, "#e1f3ec", C["green"], "Hierarchical\n4 groups→fusion", fs=16)
    ax.text(6.55, 0.55, "single coin / pooled · 1h, 4h, 1d", ha="center", fontsize=16, style="italic")
    _arrow(ax, (8.6, 2.45), (9.2, 2.45))
    # ---- right: effect ladder (medians from tables/fig_ladder_summary.csv)
    ax.text(11.15, 4.85, "What lowers the error?", **hdr)
    lad = pd.read_csv(OUT_T / "fig_ladder_summary.csv", index_col=0)["median"]
    names = {"Pooling across coins": "Pooling coins", "Hierarchy (single coin)": "Hierarchy, single coin",
             "Calendar inputs": "Calendar inputs", "Market data (intraday)": "Exchange market data",
             "Hierarchy (pooled)": "Hierarchy, pooled", "Quaternion algebra": "Quaternion algebra",
             "Social + sentiment data": "Social data", "Fusion operator": "Fusion operator"}
    for i, (k, v) in enumerate(lad.items()):
        y0 = 4.2 - i * 0.44
        col = C["blue"] if v < -2 else C["grey"]
        ax.text(11.0, y0, names[k], ha="right", va="center", fontsize=16)
        ax.add_patch(plt.Rectangle((11.15, y0 - 0.15), max(abs(v), 0.15) * 0.12, 0.30, color=col))
        ax.text(11.15 + max(abs(v), 0.15) * 0.12 + 0.08, y0, f"{v:+.1f}%".replace("-", "\u2212"), va="center",
                fontsize=15, color=col)
    ax.text(11.0, 0.45, "Median MSE change over model pairs.\nBest network (4h): 0.94 × interacted seasonal HAR,\n≈ pooled LightGBM with lags (0.99).", ha="center",
            va="center", fontsize=14.5, style="italic", color="#1f3b57")
    fig.savefig(OUT_F / "graphical_abstract.png", dpi=100)
    fig.savefig(OUT_F / "graphical_abstract.pdf")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"forest": fig_forest, "ladder": fig_ladder, "equity": fig_equity, "ga": fig_ga}
    for k, fn in jobs.items():
        if what in (k, "all"):
            fn()
