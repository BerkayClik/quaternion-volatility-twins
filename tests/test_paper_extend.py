"""Exchange-only samples that continue past the end of the vendor data (make_dataset(extend_past_lunar=True))."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib")
from src.paper.data import FREQ, GROUPS, LUNAR_GROUPS, make_dataset  # noqa: E402


def _write(root, n=900, n_vendor=700):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="time")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    prev = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(close, prev) * (1 + rng.uniform(0.001, 0.01, n))
    low = np.minimum(close, prev) * (1 - rng.uniform(0.001, 0.01, n))
    vol = rng.uniform(50, 150, n)
    (root / "binance").mkdir(parents=True)
    (root / "lunarcrush").mkdir()
    pd.DataFrame({"open": prev, "high": high, "low": low, "close": close, "volume": vol, "quote_volume": vol * close,
                  "trades": rng.integers(100, 1000, n), "taker_buy_base": vol * rng.uniform(0.3, 0.7, n)}, index=idx
                 ).to_csv(root / "binance" / "tst_4h.csv")
    cols = ["contributors_active", "contributors_created", "posts_active", "posts_created", "sentiment", "galaxy_score",
            "social_dominance", "interactions"]
    pd.DataFrame(rng.uniform(10, 90, (n_vendor, len(cols))), index=idx[:n_vendor], columns=cols
                 ).to_csv(root / "lunarcrush" / f"tst_{FREQ['4h'][0]}.csv")
    return idx


def test_extended_sample_keeps_the_common_sample_and_continues(tmp_path):
    idx = _write(tmp_path)
    common = make_dataset("tst", "4h", root=str(tmp_path)).frame
    longer = make_dataset("tst", "4h", root=str(tmp_path), extend_past_lunar=True).frame
    assert common.index[-1] == idx[699]                               # the common sample ends with the vendor data
    assert longer.index[:len(common)].equals(common.index)            # same rows up to that end
    assert longer.index[-1] == idx[-2]                                # the last bar has no next-bar target
    assert len(longer) == len(common) + (len(idx) - 1 - 700)
    exchange = [c for g in ("price", "price_r", "market", "technical", "calendar") for c in GROUPS[g]]
    assert not longer[exchange + ["y_range"]].isna().any().any()
    np.testing.assert_allclose(longer.loc[common.index, exchange].to_numpy(), common[exchange].to_numpy())
    vendor = [c for g in LUNAR_GROUPS for c in GROUPS[g]]
    assert longer[vendor].iloc[-50:].isna().all().all()               # no vendor value is carried past its end


def test_default_sample_is_unchanged(tmp_path):
    _write(tmp_path)
    a = make_dataset("tst", "4h", root=str(tmp_path)).frame
    b = make_dataset("tst", "4h", root=str(tmp_path), extend_past_lunar=False).frame
    pd.testing.assert_frame_equal(a, b)
