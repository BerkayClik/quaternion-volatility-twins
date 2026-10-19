"""
Download spot OHLCV klines from Binance's public market-data API (no key).

LunarCrush's own OHLC fields are unreliable (daily close == high on ~97% of
BTC bars, some bars with high < max(open, close)), so price/volume channels
come from Binance and LunarCrush is used only for social/sentiment fields.

Output: data/binance/{coin}_{interval}.csv indexed by bar *open* time (UTC),
columns: open, high, low, close, volume, quote_volume, trades, taker_buy_base.

Usage:
    python scripts/download_binance.py                       # 5 coins x 1d,4h
    python scripts/download_binance.py --coin btc --interval 4h
"""

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
COINS = ["btc", "eth", "bnb", "sol", "xrp", "ada", "doge", "link", "ltc", "dot",
         "avax", "trx", "bch", "xlm", "atom"]
INTERVALS = ["1d", "4h", "1h"]
START = "2017-01-01"
LIMIT = 1000

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]


def fetch_klines(symbol: str, interval: str, start_ms: int) -> list:
    rows = []
    while True:
        url = f"{BASE_URL}?symbol={symbol}&interval={interval}&startTime={start_ms}&limit={LIMIT}"
        with urlopen(url, timeout=30) as resp:
            batch = json.loads(resp.read())
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        if len(batch) < LIMIT:
            break
        time.sleep(0.2)
    return rows


def download(coin: str, interval: str, out_dir: Path, start: str = START, end: str | None = None) -> Path:
    symbol = f"{coin.upper()}USDT"
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    df = pd.DataFrame(fetch_klines(symbol, interval, start_ms), columns=COLUMNS)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.index.name = "datetime"
    df = df[KEEP].astype(float)

    # Keep only bars closed before `end` (default: now), i.e. drop the still-forming bar.
    bar = pd.Timedelta(interval.replace("d", "D"))
    cutoff = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    df = df[df.index + bar <= cutoff]
    df = df[~df.index.duplicated(keep="first")].sort_index()

    out = out_dir / f"{coin}_{interval}.csv"
    df.to_csv(out)
    print(f"{symbol} {interval}: {len(df)} bars {df.index[0]} -> {df.index[-1]} -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--coin", choices=COINS)
    parser.add_argument("--interval", choices=INTERVALS)
    parser.add_argument("--out-dir", default="data/binance")
    parser.add_argument("--start", default=START, help="first bar (UTC date)")
    parser.add_argument("--end", help="keep bars closed by this UTC date (default: now)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for coin in [args.coin] if args.coin else COINS:
        for interval in [args.interval] if args.interval else INTERVALS:
            download(coin, interval, out_dir, args.start, args.end)


if __name__ == "__main__":
    main()
