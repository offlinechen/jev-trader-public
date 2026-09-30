"""Bulk OHLCV download from the Binance monthly archives.

One code path for every timeframe: the 1m bars the label engine needs and the
15m bars the feature engine needs come from the same endpoint. 1h and 4h
context is resampled from 15m inside features.py rather than downloaded --
the grids align exactly, so a separate download would only add a way for the
two to disagree.

Archives are ~3 orders of magnitude faster than paging the REST API: three
years of 1m is ~1.5M bars per symbol.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"

RAW_COLS = [
    "ts", "open", "high", "low", "close", "volume", "close_ts",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume",
    "ignore",
]
KEEP = ["ts", "open", "high", "low", "close", "volume", "taker_buy_volume"]

TF_MS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}


def months(start: str, end: str) -> list[str]:
    """Inclusive YYYY-MM range."""
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def fetch_month(symbol: str, tf: str, ym: str, timeout: int = 60) -> pd.DataFrame | None:
    """One monthly archive. None when it does not exist (future or pre-listing)."""
    url = f"{BASE}/{symbol}/{tf}/{symbol}-{tf}-{ym}.zip"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0])

    # Binance added a header row to these archives partway through 2025.
    head = raw[:64].lstrip().split(b",", 1)[0]
    skip = 0 if head.isdigit() else 1

    df = pd.read_csv(
        io.BytesIO(raw), header=None, names=RAW_COLS, skiprows=skip,
        usecols=range(len(RAW_COLS)),
    )
    # Some 2025 archives switched open_time to microseconds.
    if len(df) and df["ts"].iloc[0] > 1e15:
        df["ts"] //= 1000
    return df[KEEP].astype({"ts": "int64"})


def download(symbol: str, tf: str, start: str, end: str, data_dir: str | Path) -> Path:
    """Fetch [start, end] into data_dir/ohlcv/{symbol}_{tf}.parquet."""
    # ponytail: serial fetch, ~minutes for 44 months of 1m. Thread-pool the
    # month loop if a second symbol or a longer history makes it a bottleneck.
    # Also archives-only: the current, unfinished month has none. Research
    # does not need it and the live path does not read archives at all.
    frames, missing = [], []
    for ym in months(start, end):
        part = fetch_month(symbol, tf, ym)
        if part is None:
            missing.append(ym)
        else:
            frames.append(part)
    if not frames:
        raise RuntimeError(f"no archives found for {symbol} {tf} {start}..{end}")
    if missing:
        print(f"  no archive for: {', '.join(missing)}")

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    out = Path(data_dir) / "ohlcv" / f"{symbol}_{tf}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load(symbol: str, tf: str, data_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "ohlcv" / f"{symbol}_{tf}.parquet")


def gaps(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Missing-bar report. G1 wants zero on 15m and >=99.9% coverage on 1m."""
    step = TF_MS[tf]
    d = df["ts"].diff()
    bad = df.loc[d > step, ["ts"]].copy()
    bad["missing"] = (d[d > step] / step - 1).astype("int64")
    return bad
