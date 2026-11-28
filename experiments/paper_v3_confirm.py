"""
Confirmation of the models on exchange data on bars after the end of the paper's sample.

The four test blocks of the paper end on 2026-10-14 16:00 UTC, and every design choice was made with
them in view. The exchange data continue, so the models that use no vendor data can be tested on a
block that no model and no decision has seen. This file and configs/paper/v3/confirm_*.yaml were
committed before the exchange data after that date were downloaded.

PLAN (fixed in advance)
  Block      rows after 2026-10-14 16:00 UTC up to 2026-11-16 16:00 UTC: 33 days, 198 four-hour bars
             per coin, 15 coins. One further walk-forward fold with the protocol of folds 1-4
             (validation: the 180 days before the block; training: everything earlier).
  Models     candidate A  hq_gattn on price with range, market, calendar (PRMC)
             candidate B  hq_meta on price with range, market, technical, calendar (PRMTC)
             their real twins (hl_gattn, hl_meta), the first calendar design of stage 1 (hq_gattn and
             hl_gattn on PMC) and the real two-source model of the paper (hl_concat-L2+cal on PMC);
             three seeds each, forecasts averaged over the seeds.
  Benchmarks HAR, HAR with one dummy per cell of the week (HAR-S x), pooled LightGBM-S on the exchange
             features of the current bar and with their lags (three seeds each).
  Loss       MSE of the log range over all coins and bars of the block; ratios of two such losses,
             cross-coin Diebold-Mariano test, 90% interval of the ratio (as in the paper).
  Questions  1. Is each candidate below HAR and below HAR-S x?
             2. How does each candidate compare with the lagged LightGBM-S (the strongest benchmark)?
             3. How does each candidate compare with its real twin?
  The block is short (a quarter of a test block of the paper), so differences of 1-2% cannot be
  expected to be significant; the direction and size of every ratio are reported whatever they are.
  The proposed model of the paper uses vendor data and cannot be evaluated on this block.

    .venv/bin/python scripts/download_binance.py --interval 4h --out-dir data_post/binance
    ln -s ../data/lunarcrush data_post/lunarcrush
    .venv/bin/python -W ignore experiments/paper_v3_confirm.py check     the new download repeats the old bars
    .venv/bin/python -W ignore experiments/paper_v3_confirm.py save      benchmarks -> results/v3confirm/bench
    .venv/bin/python experiments/paper_run.py --config configs/paper/v3/confirm_<name>_pool_4h.yaml \\
        --out results/v3confirm/out --data-root data_post --device cpu --threads 1
    .venv/bin/python -W ignore experiments/paper_v3_confirm.py eval      tables/v3_confirm.csv
    .venv/bin/python -W ignore experiments/paper_v3_confirm.py test      tables/v3_test_blocks.csv (the four
                                                                         test blocks of the paper, not a confirmation)
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = "data_post"
END, TEST_DAYS, VAL_DAYS, WINDOW, FREQ = "2026-11-16 16:00", 33, 180, 42, "4h"
OLD_END = pd.Timestamp("2026-10-14 16:00", tz="UTC")
COINS = ["btc", "eth", "sol", "bnb", "xrp", "ada", "doge", "link", "ltc", "dot", "avax", "trx", "bch", "xlm", "atom"]
BENCH_DIR = "results/v3confirm/bench"
OUT = "docs/paper/manuscript_v2/tables/v3_confirm.csv"

HAR = ("P", "har")
MODELS = [   # (label, key)
    ("HAR", HAR), ("HAR-S x (week cells)", ("P", "hars_x")),
    ("LightGBM-S, pooled, exchange features", ("PM", "lgbms-pool")),
    ("LightGBM-S, pooled, lagged exchange features", ("PM", "lgbmsl-pool")),
    ("H-LSTM, price and market, calendar term (paper)", ("PMC", "hl_concat-L2+cal-pool")),
    ("H-LSTM, first calendar design", ("PMC", "hl_gattn-pool")),
    ("H-QLSTM, first calendar design", ("PMC", "hq_gattn-pool")),
    ("Candidate A, real twin", ("PRMC", "hl_gattn-pool")),
    ("Candidate A (H-QLSTM, attention fusion, range input)", ("PRMC", "hq_gattn-pool")),
    ("Candidate B, real twin", ("PRMTC", "hl_meta-pool")),
    ("Candidate B (H-QLSTM, meta-quaternion, exchange groups)", ("PRMTC", "hq_meta-pool")),
]
CANDIDATES = {"A": (("PRMC", "hq_gattn-pool"), ("PRMC", "hl_gattn-pool")), "B": (("PRMTC", "hq_meta-pool"), ("PRMTC", "hl_meta-pool"))}


def frames() -> dict:
    """Datasets and the confirmation fold, exactly as the runner builds them from the v3 confirm configs."""
    from src.paper.data import make_dataset, walk_forward_folds
    out = {}
    for coin in COINS:
        d = make_dataset(coin, FREQ, root=ROOT, extend_past_lunar=True)
        (fold,) = walk_forward_folds(d.frame.index, 1, TEST_DAYS, VAL_DAYS, end=END)
        assert d.frame.index[fold["test"][0]] > OLD_END, "the block must start after the paper's sample"
        out[f"{coin}_{FREQ}"] = (d, fold)
    return out


def check():
    """The new download must repeat the bars of the paper's snapshot: same rows and features up to the old end."""
    from src.paper.data import GROUPS, make_dataset
    cols = [c for g in ("price_r", "market", "technical", "calendar") for c in GROUPS[g]] + ["y_range"]
    for coin in COINS:
        old = make_dataset(coin, FREQ).frame
        new = make_dataset(coin, FREQ, root=ROOT, extend_past_lunar=True).frame
        head = new[new.index <= old.index[-1]]
        assert head.index.equals(old.index), f"{coin}: rows differ in the old period"
        gap = float(np.abs(head[cols].to_numpy() - old[cols].to_numpy()).max())
        extra = new[new.index > old.index[-1]]
        print(f"{coin}: old period identical to {gap:.1e}; {len(extra)} new rows, {extra.index[0]} to {extra.index[-1]}")
        assert gap < 1e-9, f"{coin}: features differ in the old period by {gap}"


def _save(name: str, ds: str, gs: str, seed: int, frame: pd.DataFrame, res: dict):
    idx = frame.index if frame.index.tz is None else frame.index.tz_convert(None)
    t = idx.as_unit("ns").asi8
    y = frame["y_range"].to_numpy(float)
    path = os.path.join(BENCH_DIR, "preds", f"{ds}__range__{gs}__{name}__s{seed}__f0.npz")
    np.savez(path, test_time=t[res["test_rows"]], test_y=y[res["test_rows"]].astype("float32"),
             test_pred=np.asarray(res["test_pred"], "float32"), val_time=t[res["val_rows"]],
             val_y=y[res["val_rows"]].astype("float32"), val_pred=np.asarray(res["val_pred"], "float32"),
             meta=np.array(json.dumps(dict(freq=FREQ, seed=seed, fold=0, model=name))))


def save(seeds=(0, 1, 2)):
    """Benchmarks on the confirmation fold, fitted as in the paper (HAR and HAR-S x per coin by OLS on the
    training block; LightGBM-S pooled over the coins with early stopping on the validation block)."""
    import lightgbm  # noqa: F401  (its OpenMP runtime first, as in the runner)
    from paper_round7_checks import _interactions
    from src.paper.baselines import HAR_LAGS, _har_design, _ols, predict_baseline, predict_baseline_pooled
    os.makedirs(os.path.join(BENCH_DIR, "preds"), exist_ok=True)
    json.dump({"config": {"name": "v3_confirm_bench", "note": "paper_v3_confirm.save"}}, open(os.path.join(BENCH_DIR, "config.json"), "w"), indent=1)
    fr = frames()
    for ds, (d, fold) in fr.items():
        _save("har", ds, "P", 0, d.frame, predict_baseline("har", d.frame, "range", fold, WINDOW, FREQ))
        y = d.frame["y_range"].to_numpy(float)
        D = np.column_stack([_har_design(d.frame["log_range"].to_numpy(float), HAR_LAGS[FREQ]), _interactions(d.frame.index, d.step)])
        ok = fold["train"][~np.isnan(D[fold["train"]]).any(1)]
        b = _ols(D[ok], y[ok])
        te, va = (r[r >= WINDOW - 1] for r in (fold["test"], fold["val"]))
        _save("hars_x", ds, "P", 0, d.frame, dict(test_rows=te, test_pred=b[0] + D[te] @ b[1:], val_rows=va, val_pred=b[0] + D[va] @ b[1:]))
    dss = list(fr)
    Xs = [fr[ds][0].X(["price", "market"]) for ds in dss]
    Xl = [np.column_stack([X] + [pd.DataFrame(X).rolling(k).mean().to_numpy() for k in HAR_LAGS[FREQ][1:]]) for X in Xs]
    for name, X in (("lgbms-pool", Xs), ("lgbmsl-pool", Xl)):
        for seed in seeds:
            res = predict_baseline_pooled("lgbms", [fr[ds][0].frame for ds in dss], "range", [fr[ds][1] for ds in dss], WINDOW, FREQ, X, seed=seed)
            for ds, r in zip(dss, res):
                _save(name, ds, "PM", seed, fr[ds][0].frame, r)
            print(name, "seed", seed, flush=True)
    print("saved", BENCH_DIR)


def evaluate() -> pd.DataFrame:
    import paper_v2_round2 as R
    from src.paper.evaluate import load_run
    parts = [load_run(d) for d in [BENCH_DIR] + sorted(glob.glob("results/v3confirm/out/v3_confirm_*"))]
    df = pd.concat(parts, ignore_index=True)
    df = df[(df.task == "range") & (df.seed <= 2)]
    assert df.time.min() > OLD_END, "a forecast lies inside the paper's sample"
    st = R.Store([df])
    cells = df.groupby(["groupset", "model"]).apply(lambda g: len(g[["dataset", "seed"]].drop_duplicates()), include_groups=False)
    rows = []
    for label, key in MODELS:
        if not st.has(key):
            print("not available:", label)
            continue
        r = dict(model=label, fits=int(cells.loc[key]), mse_har=st.vs_har(key))
        if key != HAR:
            v = st.pair(key, HAR)
            r.update(better_than_har=v["better"], p_har=v["p"])
        lag = ("PM", "lgbmsl-pool")
        if key not in (HAR, lag) and st.has(lag):
            v = st.pair(key, lag)
            r.update(vs_lagged_lgbm=v["ratio"], lag_lo=v["ci_lo"], lag_hi=v["ci_hi"], lag_better=v["better"], p_lag=v["p"])
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    print(t.round(4).to_string(index=False))
    for name, (q, r_) in CANDIDATES.items():
        for other, what in ((r_, "its real twin"), (("P", "hars_x"), "HAR-S x"), (("PM", "lgbms-pool"), "LightGBM-S"), (("PMC", "hq_gattn-pool"), "H-QLSTM, first calendar design")):
            if st.has(q, other):
                v = st.pair(q, other)
                print(f"candidate {name} / {what}: {v['ratio']:.4f} ({v['ci_lo']:.4f}-{v['ci_hi']:.4f}), better on {v['better']}/15 coins, p = {v['p']:.4f}")
    n = df[(df.groupset == "P") & (df.model == "har")].groupby("dataset").size()
    print(f"block: {df.time.min()} to {df.time.max()}, {int(n.median())} bars per coin, {len(n)} coins")
    return t


TEST_RUNS = ["results/final_fa/out/final_single_4h_*", "results/final_fc/out/final_calendar_pool_4h_*",
             "results/v2cal/out/v2_cal_pool_4h_*", "results/r8_local/final_lgbmlags_4h_r8", "results/r8_local/final_harsx_4h_r8",
             "results/final_local/final_lgbm_4h_*", "results/v2local/v2_lgbm_seeds_4h_*",
             "results/v3local/v3_gattn_range_pool_4h_*", "results/v3local/v3_meta_exchange_pool_4h_*"]
TEST_MODELS = [("HAR", HAR), ("HAR-S x (week cells)", ("P", "hars_x")), ("LightGBM-S, pooled (16 features)", ("PMSE", "lgbms-pool")),
               ("LightGBM-S, pooled, lagged exchange features", ("PM", "lgbmsl-pool")),
               ("Proposed model of the paper (four sources, meta-quaternion)", ("PMSEC", "hq_meta-L2+cal-pool")),
               ("Its real twin", ("PMSEC", "hl_meta-L2+cal-pool")),
               ("H-LSTM, first calendar design", ("PMC", "hl_gattn-pool")), ("H-QLSTM, first calendar design", ("PMC", "hq_gattn-pool")),
               ("Candidate A, real twin", ("PRMC", "hl_gattn-pool")), ("Candidate A", ("PRMC", "hq_gattn-pool")),
               ("Candidate B, real twin", ("PRMTC", "hl_meta-pool")), ("Candidate B", ("PRMTC", "hq_meta-pool"))]


def test_blocks() -> pd.DataFrame:
    """The candidates on the four test blocks of the paper (blocks that the choice of the candidates has seen):
    MSE relative to HAR, each model against the lagged LightGBM-S, and the model confidence set (90%)."""
    import paper_v2_round2 as R
    import paper_v2_summary as S
    from arch.bootstrap import MCS
    keys = [k for _, k in TEST_MODELS]
    parts = []
    for pattern in TEST_RUNS:
        x = S.run(pattern)
        parts.append(x[[(g, m) in keys for g, m in zip(x.groupset, x.model)]])
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["dataset", "groupset", "model", "seed", "fold", "time"], keep="last")
    st = R.Store([df])
    cols = {}
    for k in keys:
        e = st.by[k].groupby(["dataset", "time"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
        cols[k] = e.assign(se=(e.pred - e.y) ** 2).pivot(index="time", columns="dataset", values="se").dropna().mean(axis=1)
    L = pd.DataFrame(cols).dropna()
    m = MCS(L.to_numpy(), size=0.10, reps=2000, block_size=max(2, int(len(L) ** (1 / 3))), seed=0)
    m.compute()
    pv = m.pvalues.squeeze()
    lag = ("PM", "lgbmsl-pool")
    rows = []
    for i, (label, key) in enumerate(TEST_MODELS):
        r = dict(model=label, mse_har=st.vs_har(key), mcs_p=float(pv.loc[i]))
        if key not in (HAR, lag):
            v = st.pair(key, lag)
            r.update(vs_lagged_lgbm=v["ratio"], lag_lo=v["ci_lo"], lag_hi=v["ci_hi"], lag_better=v["better"], p_lag=v["p"])
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT.replace("v3_confirm", "v3_test_blocks"), index=False)
    print(t.round(4).to_string(index=False))
    for name, (q, r_) in CANDIDATES.items():
        v = st.pair(q, r_)
        print(f"candidate {name} / its real twin: {v['ratio']:.4f} ({v['ci_lo']:.4f}-{v['ci_hi']:.4f}), better on {v['better']}/15 coins, p = {v['p']:.4f}, equivalent at 1%: {v['equivalent']}")
    return t


FIG = "docs/paper/review/advisor_2026-11-17/candidate_a/candidate_A_results.png"
# rows of the chart: (label, name in v3_test_blocks.csv or None, name in v3_confirm.csv or None, note if missing)
FIG_ROWS = [
    ("Candidate A (H-QLSTM)", "Candidate A", "Candidate A (H-QLSTM, attention fusion, range input)", ""),
    ("Candidate A, real twin (H-LSTM)", "Candidate A, real twin", "Candidate A, real twin", ""),
    ("LightGBM-S, lagged features", "LightGBM-S, pooled, lagged exchange features", "LightGBM-S, pooled, lagged exchange features", ""),
    ("Candidate B (meta-quaternion)", "Candidate B", "Candidate B (H-QLSTM, meta-quaternion, exchange groups)", ""),
    ("Proposed model of the manuscript", "Proposed model of the paper (four sources, meta-quaternion)", None, "not evaluated: needs vendor data"),
    ("LightGBM-S", "bench:LightGBM-S, pooled, exchange features", "LightGBM-S, pooled, exchange features", ""),
    ("Seasonal HAR (week cells)", "HAR-S x (week cells)", "HAR-S x (week cells)", ""),
    ("QCNN, 6 convolutions", "main:QCNN, 6 convolutions", None, "not run on the new data"),
]


def fig():
    """Chart for the advisor: reduction of the log-range MSE relative to HAR (per cent) of candidate A and the
    main comparators, on the four test blocks of the manuscript and on the confirmation block. Read from
    tables/v3_test_blocks.csv, v3_confirm.csv and v2_main.csv. One accent colour for candidate A, neutral grey for
    the context (the model names and the printed values carry the identity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = os.path.dirname(OUT)
    test = pd.read_csv(os.path.join(T, "v3_test_blocks.csv")).set_index("model").mse_har
    conf = pd.read_csv(os.path.join(T, "v3_confirm.csv")).set_index("model").mse_har
    main = pd.read_csv(os.path.join(T, "v2_main.csv")).set_index("label").mse
    bench = pd.read_csv(os.path.join(T, "v2_bench.csv")).set_index("model").mse_har

    def val(series, name):
        if name is None:
            return None
        for prefix, table in (("main:", main), ("bench:", bench)):
            if name.startswith(prefix):
                return float(table[name[len(prefix):]])
        return float(series[name])
    rows = [(label, val(test, a), val(conf, b), note) for label, a, b, note in FIG_ROWS]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    ACCENT, GREY, INK, MUTED = "#0072B2", "#8C8C8C", "#1a1a1a", "#6b6b6b"
    fig_, axes = plt.subplots(1, 2, figsize=(9.6, 3.75), sharey=True)
    titles = ["Four test blocks of the manuscript\n22 Apr 2025 - 14 Oct 2026 (540 days)",
              "New data, never used before\n15 Oct - 16 Nov 2026 (33 days)"]
    y = list(range(len(rows)))[::-1]
    for ax, col, title in zip(axes, (1, 2), titles):
        for yi, row in zip(y, rows):
            ratio = row[col]
            if ratio is None:
                ax.text(0.4, yi, row[3], va="center", ha="left", fontsize=8, color=MUTED, style="italic")
                continue
            gain = 100 * (1 - ratio)
            first = row[0].startswith("Candidate A (")
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
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    fig_.suptitle("Candidate A against the strongest benchmarks: error of the next-bar log-range forecast",
                  x=0.012, y=0.975, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig_.text(0.012, 0.012, "15 cryptocurrencies, 4-hour bars, pooled training, three-seed averages; LightGBM-S on exchange features. "
              "The models of the right panel were fixed before the new data were downloaded.", fontsize=7.5, color=MUTED, ha="left")
    fig_.tight_layout(rect=(0, 0.035, 1, 0.985), w_pad=1.6)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig_.savefig(FIG, dpi=220, facecolor="white")
    plt.close(fig_)
    print("written", FIG)
    for r in rows:
        print(f"  {r[0]:36s} test {'' if r[1] is None else round(100 * (1 - r[1]), 1)}  new {'' if r[2] is None else round(100 * (1 - r[2]), 1)}")


if __name__ == "__main__":
    {"check": check, "save": save, "eval": evaluate, "test": test_blocks, "fig": fig}[sys.argv[1]]()
