"""
Block after the sample for manuscript v5: point 3 of the plan of experiments/paper_v4_round3.py (2026-11-24),
"one fold on all four-hour bars after 2026-10-14 16:00 UTC up to the last complete bar at the download, for both
sets of coins, with the models, seeds and evaluation of point 2".

Plan (2026-11-28, written and committed before the bars after 2026-11-16 are downloaded and before any model of
this block is fitted).

  block      Rows after 2026-10-14 16:00 UTC up to 2026-11-28 00:00 UTC: 44 days and 8 hours, 266 four-hour bars
             per coin, 3,990 forecasts for each set of 15 coins if no bar is missing. The target of the last row is
             the bar from 04:00 to 08:00 UTC of 2026-11-28, the last complete bar at the download (made between
             08:00 and 12:00 UTC of that day; an earlier version of this plan, commit 44c0bc3, named one bar more
             and a download after 12:00 UTC; changed before anything was downloaded). One walk-forward fold with the protocol of the paper: validation on the 180 days before the block, training
             on everything earlier, samples from 2020-01-01 or the listing, exchange data only (`exchange_only`).
  coins      the 15 coins of the study and the 15 new coins of paper_v4_newcoins (configs/paper/v4/newcoins.json).
  models     hq_gattn, hl_gattn and the Transformer on price with range, market and calendar, 20,000 parameters,
             window 42, seeds 0-4, the training protocol of the paper, nothing tuned
             (configs/paper/v4/post_{study,new}_pool_4h.yaml); the lagged LightGBM-S (seeds 0-4), HAR and HAR-S x.
  evaluation paper_v4_newcoins.evaluate, unchanged: tables/v4_post_{study,new}.csv and *_limits.csv (MSE, QLIKE, the
             two tests and the rule of paper_v3_tests, 90% interval of every difference, tick loss of the 95% range
             limits). And the comparisons of the quaternion with the real version, network and hybrid, by the routine
             of paper_v5_stats.twin (tables/v5_post_twin.csv).
  known      For the 15 coins of the study the first 198 of the 266 bars (to 2026-11-16) were evaluated on 2026-11-17
             with other fits (three runs, samples that a vendor feed limits): the quaternion hybrid had 0.997 of the
             MSE of LightGBM (90% interval 0.981-1.013) and was 2.5% below its network (tables/v3_confirm.csv). The
             block is therefore new for these coins on its last 68 bars only, and the text says so. For the 15 new
             coins no bar after 2026-10-14 was downloaded or evaluated before.
  changes    against point 3 as written on 2026-11-24, both made before any number:
             (1) the block is run on 2026-11-28, not on the day before submission, by decision of the author; it
                 holds 44 days instead of about 50. It is run once: it is not repeated or extended before
                 submission, whatever it shows.
             (2) "the hybrid" of that plan was the hybrid with real layers. Since the decision of 2026-11-27 the
                 proposed system is the quaternion hybrid, and the text reports the block for it (rows with
                 ours = "H-QLSTM + LightGBM"); the hybrid with the real twin is reported next to it.
  reading    fixed here. For each set of coins the text gives the MSE of the hybrid relative to HAR, its difference
             to LightGBM with the 90% interval and the two p-values, and its difference to its network. A set counts
             as "confirmed on the block" only if the hybrid is better than LightGBM and than its network by the
             two-test rule; otherwise the text says that the block does not establish the difference and gives the
             interval. The abstract states the outcome in one clause. The block is not added to the three held-out
             data sets of the paper and changes none of their statements. Reported whatever it shows.

    .venv/bin/python -W ignore experiments/paper_v5_post.py download            30 coins -> data_sub/binance
    .venv/bin/python -W ignore experiments/paper_v5_post.py check               the new download repeats the old bars
    .venv/bin/python -W ignore experiments/paper_v5_post.py save [study|new]    benchmarks -> results/v5post/bench_<set>
    (networks: configs/paper/v4/post_{study,new}_pool_4h.yaml on Colab, bundle built with DATA_SRC=data_sub
     -> results/v5post/out)
    .venv/bin/python -W ignore experiments/paper_v5_post.py eval [study|new]    tables/v4_post_<set>.csv, *_limits.csv
    .venv/bin/python -W ignore experiments/paper_v5_post.py twin                tables/v5_post_twin.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = "data_sub"
OLD_END = pd.Timestamp("2026-10-14 16:00", tz="UTC")
END, TEST_DAYS, VAL_DAYS = "2026-11-28 00:00", 133 / 3, 180      # 44 days and 8 hours: 266 bars
CSV = "docs/paper/manuscript_v2/tables"
BAR = pd.Timedelta(hours=4)


def _coins(tag: str) -> list[str]:
    import paper_v4_newcoins as N
    return list(N.OLD) if tag == "study" else json.load(open(N.UNIVERSE))["chosen"]


def _post(tag: str):
    """paper_v4_newcoins with the fold after the sample, one of the two sets of coins and the new download."""
    import importlib

    import paper_v4_newcoins as N
    from src.paper.data import make_dataset, walk_forward_folds
    importlib.reload(N)
    coins = _coins(tag)
    N.chosen = lambda: coins
    N.TAG, N.FOLDS = f"post_{tag}", dict(n=1, test_days=TEST_DAYS, val_days=VAL_DAYS, end=END)
    N.BENCH, N.OUT_RUNS = f"results/v5post/bench_{tag}", f"results/v5post/out/v4_post_{tag}_pool_4h_*"

    def frames() -> dict:
        out = {}
        for coin in coins:
            d = make_dataset(coin, N.FREQ, root=ROOT, start=N.START, exchange_only=True)
            folds = walk_forward_folds(d.frame.index, 1, TEST_DAYS, VAL_DAYS, end=END)
            assert folds[0]["test_start"] == OLD_END and len(folds[0]["test"]) > 0, coin
            out[f"{coin}_{N.FREQ}"] = (d, folds)
        return out

    N.frames = frames
    import paper_v3_runs as RU
    RU.CELLS = len(coins)            # one fold: a complete run has one cell per coin (the loader expects four folds)
    import paper_v3_tests as T3
    T3.N_TEST = 266 * len(coins)     # the guard of the loss routine expects the 48,600 forecasts of four blocks
    return N


def download():
    """Four-hour bars of the 30 coins with the downloader of the paper, up to the bar that is the last target."""
    sys.path.insert(0, "scripts")
    import download_binance as D
    last = pd.Timestamp(END, tz="UTC") + BAR
    assert pd.Timestamp.now(tz="UTC") >= last + BAR, "the last target bar is not complete yet"
    os.makedirs(f"{ROOT}/binance", exist_ok=True)
    for coin in _coins("study") + _coins("new"):
        start_ms = int(pd.Timestamp(D.START, tz="UTC").timestamp() * 1000)
        df = pd.DataFrame(D.fetch_klines(f"{coin.upper()}USDT", "4h", start_ms), columns=D.COLUMNS)
        df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.index.name = "datetime"
        df = df[D.KEEP].astype(float)
        df = df[df.index <= last]
        assert df.index[-1] == last, (coin, df.index[-1])
        df.to_csv(f"{ROOT}/binance/{coin}_4h.csv")
        print(coin, len(df), df.index[0], df.index[-1], flush=True)


def check():
    """The new download must repeat the bars that the paper used: same rows, features and targets up to the old end."""
    from src.paper.data import GROUPS, make_dataset
    cols = [c for g in ("price_r", "market", "calendar") for c in GROUPS[g]] + ["y_range"]
    for tag in ("study", "new"):
        for coin in _coins(tag):
            old = make_dataset(coin, "4h", start="2020-01-01", exchange_only=True).frame
            new = make_dataset(coin, "4h", root=ROOT, start="2020-01-01", exchange_only=True).frame
            old, head, extra = old[old.index <= OLD_END], new[new.index <= OLD_END], new[new.index > OLD_END]
            assert head.index.equals(old.index), f"{coin}: rows differ in the old period"
            gap = float(np.abs(head[cols].to_numpy() - old[cols].to_numpy()).max())
            print(f"{tag} {coin}: old period identical to {gap:.1e}; {len(extra)} new rows, {extra.index[0]} to {extra.index[-1]}")
            assert gap < 1e-9, f"{coin}: features differ in the old period by {gap}"
            assert len(extra) == 266 and extra.index[-1] == pd.Timestamp(END, tz="UTC"), (coin, len(extra))


def twin():
    """The quaternion version against the real one, network and hybrid, on the two blocks (routine of paper_v5_stats.twin)."""
    import paper_v3_runs as RU
    import paper_v4_round4 as R4
    mix = lambda *fs: fs[0].assign(pred=sum(f.reindex(fs[0].index).pred for f in fs) / len(fs))
    rows = []
    for tag in ("study", "new"):
        N = _post(tag)
        caps = pd.DataFrame([dict(dataset=ds, fold=k, cap=float(np.exp(2 * np.nanmax(d.frame["y_range"].to_numpy(float)[fo["train"]]))))
                             for ds, (d, folds) in N.frames().items() for k, fo in enumerate(folds)]).set_index(["dataset", "fold"]).cap
        get = RU.forecasts
        L, Q, R = get(("PM", "lgbmsl-pool"), [N.BENCH]), get(("PRMC", "hq_gattn-pool"), [N.OUT_RUNS]), get(("PRMC", "hl_gattn-pool"), [N.OUT_RUNS])
        assert all(sorted(m) == list(N.SEEDS) for m in (L, Q, R)), tag
        X = {"H-QLSTM": Q, "H-LSTM": R, "H-QLSTM + LightGBM": {k: mix(Q[k], L[k]) for k in N.SEEDS}, "H-LSTM + LightGBM": {k: mix(R[k], L[k]) for k in N.SEEDS}}
        B = R4._bars(X, caps)
        for what, a, b in (("network", "H-QLSTM", "H-LSTM"), ("hybrid", "H-QLSTM + LightGBM", "H-LSTM + LightGBM")):
            rows.append(dict(data=f"post_{tag}", freq="4h", what=what, **R4._cmp(B, a, b)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(CSV, "v5_post_twin.csv"), index=False)
    pd.set_option("display.width", 280, "display.max_columns", 40)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "download":
        download()
    elif cmd == "check":
        check()
    elif cmd == "save":
        _post(rest[0]).save()
    elif cmd == "eval":
        _post(rest[0]).evaluate()
    elif cmd == "twin":
        twin()
