#!/usr/bin/env python3
"""
Download LunarCrush coin time series (API v4, time-series/v2) for the paper.

One request per coin, bucket and calendar year (the API returns a full year of
hourly rows in one call). Raw yearly responses are cached as JSON, so reruns
cost no API tokens. Output, bars indexed by open time (UTC):

    data/lunarcrush/raw/{coin}_{bucket}_{year}.json
    data/lunarcrush/{coin}_day.csv, {coin}_hour.csv
    data/lunarcrush/{coin}_4hour.csv   (resampled from hourly, see AGG_RULES)

Only bars that closed before --end are kept (no partial last bar).

    python scripts/download_lunarcrush_paper.py --coins btc,eth --start 2020-01-01 --end 2026-10-15
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

API = "https://lunarcrush.com/api4/public/coins/{coin}/time-series/v2"
OUT = Path("data/lunarcrush")
STEP = {"hour": pd.Timedelta("1h"), "day": pd.Timedelta("1D")}

# Hourly -> 4-hourly aggregation (same rules as scripts/resample_4hourly.py).
AGG_RULES = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume_24h": "last", "market_cap": "last", "circulating_supply": "last",
    "market_dominance": "last", "alt_rank": "last",
    "sentiment": "mean", "galaxy_score": "mean", "social_dominance": "mean",
    "interactions": "sum", "contributors_created": "sum", "posts_created": "sum", "spam": "sum",
    "contributors_active": "max", "posts_active": "max",
}


def api_key() -> str:
    if os.getenv("LUNARCRUSH_API_KEY"):
        return os.environ["LUNARCRUSH_API_KEY"]
    for line in open(".env"):
        if line.startswith("LUNARCRUSH_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("LUNARCRUSH_API_KEY not set")


def fetch(coin: str, bucket: str, start: pd.Timestamp, end: pd.Timestamp, key: str) -> list[dict]:
    url = f"{API.format(coin=coin)}?bucket={bucket}&start={int(start.timestamp())}&end={int(end.timestamp())}"
    for attempt in range(5):
        res = subprocess.run(["curl", "-s", "-w", "\n%{http_code}", "-H", f"Authorization: Bearer {key}", url],
                             capture_output=True, text=True)
        body, code = res.stdout.rsplit("\n", 1)
        time.sleep(1.5)
        if code == "200":
            return json.loads(body)["data"]
        print(f"    HTTP {code}, retry {attempt + 1}", flush=True)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed: {coin} {bucket} {start.date()}")


def download(coin: str, bucket: str, start: pd.Timestamp, end: pd.Timestamp, key: str) -> pd.DataFrame:
    rows = []
    for year in range(start.year, end.year + 1):
        cache = OUT / "raw" / f"{coin}_{bucket}_{year}.json"
        y0 = max(start, pd.Timestamp(f"{year}-01-01", tz="UTC"))
        y1 = min(end, pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        complete = y1 == pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            data = fetch(coin, bucket, y0, y1, key)
            if complete:  # never cache a partial current year under a final name
                cache.write_text(json.dumps(data))
        rows += data
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True).rename("Datetime")
    df = df.apply(pd.to_numeric, errors="coerce")  # a few values arrive as strings
    df = df[~df.index.duplicated()].sort_index()
    return df[(df.index >= start) & (df.index + STEP[bucket] <= end)]


def to_4h(hourly: pd.DataFrame, min_hours: int = 2) -> pd.DataFrame:
    """Aggregate hourly rows to 4h bars (labelled by open time).

    Each column is aggregated over the hours where it is present; event counts
    (`sum`) are rescaled by 4 / n_present so that a partly missing window does
    not look like a drop in activity. A value needs >= `min_hours` present hours.
    """
    r = hourly.resample("4h")
    n = r.count()
    out = {}
    for c, rule in AGG_RULES.items():
        if c not in hourly.columns:
            continue
        v = r[c].agg(rule)
        if rule == "sum":
            v = v * 4 / n[c]
        out[c] = v.where(n[c] >= min_hours)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", required=True)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-10-15")
    a = ap.parse_args()
    start, end = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    key = api_key()
    for coin in a.coins.split(","):
        for bucket in ["day", "hour"]:
            df = download(coin, bucket, start, end, key)
            df.to_csv(OUT / f"{coin}_{bucket}.csv")
            print(f"{coin} {bucket}: {len(df)} rows {df.index.min()} .. {df.index.max()}", flush=True)
            if bucket == "hour":
                h4 = to_4h(df)
                h4.to_csv(OUT / f"{coin}_4hour.csv")
                print(f"{coin} 4hour: {len(h4)} rows", flush=True)


if __name__ == "__main__":
    main()
