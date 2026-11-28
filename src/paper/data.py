"""
Dataset construction for the paper experiments.

Sources (joined on bar *open* time, UTC):
  * Binance spot klines   -> price / market / technical groups
  * LunarCrush time series -> social / sentiment groups
A row labelled t describes the bar [t, t+step) and is fully known at t+step,
which is the decision time for forecasting bar t+1.

Social data may have short outages: LunarCrush values are carried forward
for at most one day (never backwards), and each coin's sample starts after
its last longer outage, so every row has fresh social data or a <=1-day-old
value (see make_dataset).

Every feature is a stationary transform so that one train-fitted z-score is a
fair normalization for every model family (no per-window RevIN needed).
Features come in semantic groups of exactly four, i.e. one quaternion each.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import talib

FREQ = {"1d": ("day", pd.Timedelta("1D")), "4h": ("4hour", pd.Timedelta("4h")),
        "1h": ("hour", pd.Timedelta("1h"))}
SURPRISE_WINDOW = 30  # bars, for "x_t vs its recent mean" features
FFILL_LIMIT = {"1d": 1, "4h": 6, "1h": 24}  # max carried-forward LunarCrush bars (one day)

# Order inside each group = quaternion components (r, i, j, k).
GROUPS = {
    "price":     ["o_rel", "h_rel", "l_rel", "c_rel"],
    # control: the current bar's log range (the target's scale, as HAR receives it)
    # in the otherwise zero real part of the price quaternion
    "price_r":   ["log_range", "h_rel", "l_rel", "c_rel"],
    "market":    ["qvol_surprise", "trades_surprise", "taker_imbalance", "qvol_dlog"],
    "social":    ["contrib_active_s", "contrib_created_s", "posts_active_s", "posts_created_s"],
    "sentiment": ["sentiment", "galaxy_score", "social_dominance", "interactions_s"],
    "technical": ["rsi", "macd_hist", "natr", "bb_pctb"],
    # calendar of the target bar t+1 (deterministic, known at the end of bar t)
    "calendar":  ["tod_sin", "tod_cos", "dow_sin", "dow_cos"],
    # synthetic controls for the noise-robustness experiment (no information about y)
    "noise": ["noise_0", "noise_1", "noise_2", "noise_3"],
    "noise2": ["noise_4", "noise_5", "noise_6", "noise_7"],
}
# random partitions of the 16 PMSE features into 4 groups of 4 (control for semantic grouping):
# groups rand{s}_{0..3} for partition seeds s = 1, 2, 3
_PMSE = GROUPS["price"] + GROUPS["market"] + GROUPS["social"] + GROUPS["sentiment"]
for _s in (1, 2, 3):
    _perm = list(np.random.default_rng(_s).permutation(_PMSE))
    for _k in range(4):
        GROUPS[f"rand{_s}_{_k}"] = [str(c) for c in _perm[4 * _k: 4 * _k + 4]]
LUNAR_GROUPS = {"social", "sentiment"}


def _surprise(x: pd.Series) -> pd.Series:
    lx = np.log(x.clip(lower=1.0))
    return lx - lx.rolling(SURPRISE_WINDOW).mean().shift(1)


def build_features(binance: pd.DataFrame, lunar: pd.DataFrame | None, ffill_limit: int | None = None) -> pd.DataFrame:
    """All candidate features + targets, indexed by bar open time."""
    b = binance
    prev_c = b["close"].shift(1)
    f = pd.DataFrame(index=b.index)

    # price: bar shape relative to the previous close (log ratios), encoded as
    # a pure quaternion 0 + h i + l j + c k. Crypto trades 24/7, so open equals
    # the previous close and the bar has three degrees of freedom.
    f["o_rel"] = 0.0
    for col, name in [("high", "h_rel"), ("low", "l_rel"), ("close", "c_rel")]:
        f[name] = np.log(b[col] / prev_c)

    # market: Binance volume / activity
    f["qvol_surprise"] = _surprise(b["quote_volume"])
    f["trades_surprise"] = _surprise(b["trades"])
    f["taker_imbalance"] = b["taker_buy_base"] / b["volume"].replace(0, np.nan) - 0.5
    f["qvol_dlog"] = np.log(b["quote_volume"].clip(lower=1.0)).diff()

    # technical: TA-Lib indicators on Binance OHLC
    h, l, c = (b[k].to_numpy(float) for k in ("high", "low", "close"))
    f["rsi"] = talib.RSI(c, timeperiod=14) / 100.0 - 0.5
    f["macd_hist"] = talib.MACD(c, 12, 26, 9)[2] / c
    f["natr"] = talib.NATR(h, l, c, timeperiod=14) / 100.0
    upper, _, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    f["bb_pctb"] = (c - lower) / (upper - lower) - 0.5

    # social / sentiment: LunarCrush (ffill only -> no backward leakage)
    if lunar is not None:
        lu = lunar.reindex(b.index).ffill(limit=ffill_limit)
        f["contrib_active_s"] = _surprise(lu["contributors_active"])
        f["contrib_created_s"] = _surprise(lu["contributors_created"])
        f["posts_active_s"] = _surprise(lu["posts_active"])
        f["posts_created_s"] = _surprise(lu["posts_created"])
        f["sentiment"] = lu["sentiment"] / 100.0
        f["galaxy_score"] = lu["galaxy_score"] / 100.0
        f["social_dominance"] = lu["social_dominance"]
        f["interactions_s"] = _surprise(lu["interactions"])
        f["has_lunar"] = lunar.reindex(b.index).notna().any(axis=1)

    # synthetic noise groups: persistent, heavy-tailed AR(1) processes
    # (phi = 0.95, Student-t(3) innovations), independent of the market and fixed
    # per index length so every model sees the same draws.
    rng = np.random.default_rng(len(b.index))
    eps = rng.standard_t(3, size=(len(b.index), 8))
    z = np.zeros_like(eps)
    for t in range(1, len(eps)):
        z[t] = 0.95 * z[t - 1] + eps[t]
    for k in range(8):
        f[f"noise_{k}"] = z[:, k]

    # calendar of the next bar: time of day and day of week as points on the unit circle
    # (constant time of day for daily bars, which the train-fold scaler maps to zero)
    step = pd.Series(b.index).diff().median()
    nxt = b.index + step
    tod = (nxt.hour * 60 + nxt.minute) / 1440.0
    dow = (nxt.dayofweek + tod) / 7.0
    f["tod_sin"], f["tod_cos"] = np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)

    # targets for bar t+1, aligned to row t (decision at end of bar t)
    log_range = np.log(np.log(b["high"] / b["low"]).clip(lower=1e-6))
    f["y_range"] = log_range.shift(-1)          # next-bar log range
    f["y_return"] = np.log(b["close"]).diff().shift(-1)  # next-bar log return
    f["log_range"] = log_range                  # current, for baselines
    f["ret"] = np.log(b["close"]).diff()
    return f


def load_raw(coin: str, freq: str, root: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    lunar_bucket, _ = FREQ[freq]
    binance = pd.read_csv(Path(root) / "binance" / f"{coin}_{freq}.csv", index_col=0, parse_dates=True)
    lunar = pd.read_csv(Path(root) / "lunarcrush" / f"{coin}_{lunar_bucket}.csv",
                        index_col=0, parse_dates=True)
    return binance, lunar


@dataclass
class Dataset:
    """Feature matrix restricted to a common sample, plus metadata."""
    coin: str
    freq: str
    frame: pd.DataFrame          # all features + targets, clean rows only
    step: pd.Timedelta

    def X(self, groups: list[str]) -> np.ndarray:
        cols = [c for g in groups for c in GROUPS[g]]
        return self.frame[cols].to_numpy(np.float32)


def make_dataset(coin: str, freq: str, root: str = "data", require_lunar: bool = True,
                 start: str | None = None, extend_past_lunar: bool = False, exchange_only: bool = False) -> Dataset:
    """Common sample for one coin/frequency: rows where every feature and
    both targets exist (and LunarCrush data is present if required).

    `extend_past_lunar` is for models on exchange data only: the sample starts where the common
    sample starts and keeps the same rows up to the end of the LunarCrush snapshot, and it continues
    past that end for as long as the exchange features and the targets exist. The LunarCrush columns
    are missing on the added rows, so groups that use them must not be read from such a dataset.

    `exchange_only` is for coins without vendor data (the new coins of paper v4): the features are built
    from the exchange file alone, the sample is every row on which the exchange features and both targets
    exist, and the vendor columns are absent."""
    if exchange_only:
        binance = pd.read_csv(Path(root) / "binance" / f"{coin}_{freq}.csv", index_col=0, parse_dates=True)
        vendor_cols = {c for g in LUNAR_GROUPS for c in GROUPS[g]}
        exchange_cols = [c for g in GROUPS.values() for c in g if c not in vendor_cols]
        f = build_features(binance, None).replace([np.inf, -np.inf], np.nan)
        f = f.dropna(subset=exchange_cols + ["y_range", "y_return"])
        if start is not None:
            f = f[f.index >= pd.Timestamp(start, tz="UTC")]
        return Dataset(coin=coin, freq=freq, frame=f, step=FREQ[freq][1])
    binance, lunar = load_raw(coin, freq, root)
    full = build_features(binance, lunar, FFILL_LIMIT[freq])
    f = full
    feat_cols = [c for g in GROUPS.values() for c in g]
    if require_lunar:
        # LunarCrush coverage end: do not ffill past the last real observation
        f = f[f.index <= lunar.index.max()]
        # start after the last social outage longer than the ffill limit
        lunar_cols = [c for g in LUNAR_GROUPS for c in GROUPS[g]]
        gap = f[lunar_cols].isna().any(axis=1)
        if gap.any():
            f = f[f.index > gap[gap].index.max()]
    f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + ["y_range", "y_return"])
    if extend_past_lunar:
        vendor_cols = {c for g in LUNAR_GROUPS for c in GROUPS[g]}
        exchange_cols = [c for c in feat_cols if c not in vendor_cols]
        tail = full[full.index > f.index[-1]].replace([np.inf, -np.inf], np.nan)
        tail = tail.dropna(subset=exchange_cols + ["y_range", "y_return"])
        f = pd.concat([f, tail])
    if start is not None:  # features above are computed on full history, then trimmed
        f = f[f.index >= pd.Timestamp(start, tz="UTC")]
    return Dataset(coin=coin, freq=freq, frame=f, step=FREQ[freq][1])


def walk_forward_folds(index: pd.DatetimeIndex, n_folds: int, test_days: int, val_days: int,
                       end: str | None = None):
    """Expanding-window folds over row positions.

    Fold k (0 = earliest) tests on a contiguous `test_days` block; the
    `val_days` block right before it is used for early stopping; training
    uses everything earlier. The last fold ends at `end` (default: the last
    row); a shared calendar `end` aligns folds across assets for pooling.
    Returns a list of dicts of integer position arrays.
    """
    end = index[-1] if end is None else pd.Timestamp(end, tz="UTC")
    folds = []
    for k in range(n_folds):
        test_end = end - pd.Timedelta(days=test_days * (n_folds - 1 - k))
        test_start = test_end - pd.Timedelta(days=test_days)
        val_start = test_start - pd.Timedelta(days=val_days)
        pos = np.arange(len(index))
        folds.append({
            "fold": k,
            # drop each block's last row: its target bar lies in the next block
            "train": pos[index <= val_start][:-1],
            "val": pos[(index > val_start) & (index <= test_start)][:-1],
            "test": pos[(index > test_start) & (index <= test_end)],
            "test_start": test_start, "test_end": test_end,
        })
    return folds


def time_blocks(indexes: list, folds: list, blocks: int) -> list:
    """Edges of `blocks` blocks of equal length in time that together cover the training rows of all assets of
    one fold (indexes: the time index of each asset; folds: its fold). Block j holds the rows with
    edges[j] <= time < edges[j + 1]."""
    t_lo = min(idx[fo["train"][0]] for idx, fo in zip(indexes, folds))
    t_hi = max(idx[fo["train"][-1]] for idx, fo in zip(indexes, folds))
    edges = [t_lo + (t_hi - t_lo) * j / blocks for j in range(blocks + 1)]
    edges[-1] += pd.Timedelta("1s")
    return edges


def held_out_fold(index: pd.DatetimeIndex, fold: dict, edges: list, block: int) -> dict:
    """Fold for an out-of-fold forecast of training rows: block `block` of the training rows becomes the test
    block and the other training rows are trained on; the validation block is unchanged. Inputs and target
    keep the scaling of the whole training block ("scale"). An asset may have no row in the held-out block."""
    t = index[fold["train"]]
    held = np.asarray((t >= edges[block]) & (t < edges[block + 1]))
    return {"fold": fold["fold"], "train": fold["train"][~held], "val": fold["val"], "test": fold["train"][held],
            "scale": fold["train"]}
