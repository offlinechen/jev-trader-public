"""从 Binance 月度归档批量下载 OHLCV。 / Bulk OHLCV download from Binance monthly archives.

所有周期共用同一归档端点：1m 用于标签，15m 用于特征；1h/4h 从 15m 重采样，
避免独立下载导致时间网格不一致。归档比逐页 REST 请求快约三个数量级。

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
    """包含首尾月份的 YYYY-MM 区间。 / Inclusive YYYY-MM range."""
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def fetch_month(symbol: str, tf: str, ym: str, timeout: int = 60) -> pd.DataFrame | None:
    """读取单个月份归档；不存在时返回 None。 / Read one monthly archive; return None if absent."""
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

    # Binance 在 2025 年期间开始给归档加表头。 / Binance added archive headers during 2025.
    head = raw[:64].lstrip().split(b",", 1)[0]
    skip = 0 if head.isdigit() else 1

    df = pd.read_csv(
        io.BytesIO(raw), header=None, names=RAW_COLS, skiprows=skip,
        usecols=range(len(RAW_COLS)),
    )
    # 部分 2025 年归档将 open_time 改为微秒。 / Some 2025 archives use microsecond open_time.
    if len(df) and df["ts"].iloc[0] > 1e15:
        df["ts"] //= 1000
    return df[KEEP].astype({"ts": "int64"})


def download(symbol: str, tf: str, start: str, end: str, data_dir: str | Path) -> Path:
    """下载闭区间到 data_dir/ohlcv/{symbol}_{tf}.parquet。 / Fetch inclusive range to Parquet."""
    # ponytail: 串行下载 44 个月的 1m 数据约需数分钟；若更多交易对或更长历史形成瓶颈，再并行月份。
    # Serial fetch takes minutes for 44 months of 1m; parallelize months only if it becomes a bottleneck.
    # 当前未完成月份没有归档；研究不需要它，实时路径也不读归档。
    # The unfinished month has no archive; research does not need it and live code does not read archives.
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
    """缺失 K 线报告：G1 要求 15m 零缺口、1m 覆盖率至少 99.9%。 / Gap report: G1 needs no 15m gaps and >=99.9% 1m coverage."""
    step = TF_MS[tf]
    d = df["ts"].diff()
    bad = df.loc[d > step, ["ts"]].copy()
    bad["missing"] = (d[d > step] / step - 1).astype("int64")
    return bad
