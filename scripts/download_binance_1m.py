"""
1-minute Binance spot klines for the realised-variance check of the paper (review).

    .venv/bin/python scripts/download_binance_1m.py      # -> data/binance_1m/{coin}_{freq}_rv.csv

Downloads monthly (and, for the last month, daily) zip files from data.binance.vision for
the validation and test periods (Oct 2024 - 14 Oct 2026), and aggregates squared 1-minute
log returns into realised variance per bar (bar open time index, as data/binance/):
RV_t = sum over the minutes of bar [t, t+step) of (log C_m - log C_{m-1})^2.
"""

import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_binance import COINS  # noqa: E402

ROOT = Path("data/binance_1m")
BASE = "https://data.binance.vision/data/spot"
MONTHS = pd.period_range("2024-10", "2026-09", freq="M")
LAST_DAYS = pd.date_range("2026-10-01", "2026-10-14", freq="D")
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qvol", "trades", "tb_base",
        "tb_quote", "ignore"]


def fetch(url: str, dest: Path) -> Path | None:
    if dest.exists():
        return dest
    try:
        data = urlopen(url, timeout=120).read()
    except Exception as e:  # missing file (e.g. coin not listed) -> skip
        print("missing", url, e, flush=True)
        return None
    dest.write_bytes(data)
    return dest


def read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        df = pd.read_csv(io.BytesIO(z.read(name)), header=None, names=COLS, usecols=[0, 4])
    df = df[pd.to_numeric(df.open_time, errors="coerce").notna()]      # drop a header row if present
    t = df.open_time.astype("int64")
    unit = "us" if t.max() > 1e14 else "ms"                             # 2025+ files use microseconds
    return pd.DataFrame({"close": df.close.astype(float).to_numpy()},
                        index=pd.to_datetime(t.to_numpy(), unit=unit, utc=True))


def main():
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = []
    for coin in COINS:
        sym = f"{coin.upper()}USDT"
        for m in MONTHS:
            jobs.append((coin, f"{BASE}/monthly/klines/{sym}/1m/{sym}-1m-{m}.zip", raw / f"{sym}-1m-{m}.zip"))
        for d in LAST_DAYS:
            jobs.append((coin, f"{BASE}/daily/klines/{sym}/1m/{sym}-1m-{d:%Y-%m-%d}.zip",
                         raw / f"{sym}-1m-{d:%Y-%m-%d}.zip"))
    with ThreadPoolExecutor(8) as ex:
        paths = list(ex.map(lambda j: (j[0], fetch(j[1], j[2])), jobs))
    for coin in COINS:
        parts = [read_zip(p) for c, p in paths if c == coin and p is not None]
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated()]
        r2 = np.log(s.close).diff() ** 2
        for freq, rule in (("1h", "1h"), ("4h", "4h"), ("1d", "1D")):
            g = r2.resample(rule, label="left", closed="left")
            out = pd.DataFrame({"rv": g.sum(min_count=1), "n_min": g.count()})
            out.index.name = "datetime"
            out.to_csv(ROOT / f"{coin}_{freq}_rv.csv")
        print(coin, len(s), s.index.min(), s.index.max(), flush=True)


if __name__ == "__main__":
    main()
