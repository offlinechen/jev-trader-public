"""特征引擎。 / Feature engine.

每列只依赖 t 及之前的 K 线；截断历史重算测试会检查这一点。
1h/4h 从 15m 重采样，并以完整收盘时间合并，自动排除尚未收盘的高周期 K 线。

Every column produced here must be a pure function of bars at or before t.
`tests/test_causality.py` enforces it by rebuilding features on truncated
history and demanding an exact match -- read that test before adding a
feature, and never reach for `shift(-n)`, `center=True` or `bfill`.

1h and 4h context is resampled from the 15m frame rather than taken as a
separate input. The grids align exactly (4 and 16 bars), so resampling is
lossless and removes the commonest lookahead bug in this project: merging a
higher-timeframe bar that has not closed yet. Here a resampled bar carries a
close time of open + full period, so a partial trailing bar claims a close in
the future and `merge_asof(direction="backward")` excludes it automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MS = 900_000
MTF = {"1h": 4, "4h": 16}  # 15m 的倍数 / Multiples of the 15m bar.
OHLCV_FEATURES_V1 = (
    "f_ret_1", "f_ret_2", "f_ret_4", "f_ret_8", "f_ret_16", "f_ret_32",
    "f_hl_range", "f_close_pos_in_range", "f_dist_high_24h", "f_dist_low_24h",
    "f_ema20_rel", "f_ema60_rel", "f_ema120_rel", "f_ema20_slope",
    "f_ema60_slope", "f_trend_strength", "f_adx", "f_atr_pct", "f_rv_1h",
    "f_rv_4h", "f_rv_24h", "f_bb_width", "f_vol_ratio", "f_vol_z",
    "f_vol_trend", "f_1h_trend", "f_1h_vol", "f_4h_trend", "f_4h_vol",
)


def build_ohlcv_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """返回冻结的跨交易所特征契约，不含主动成交量。 / Return frozen cross-venue features without taker flow."""
    features = build_features(df.drop(columns="taker_buy_volume", errors="ignore"))
    missing = set(OHLCV_FEATURES_V1).difference(features.columns)
    if missing:
        raise ValueError(f"OHLCV feature contract is incomplete: {sorted(missing)}")
    return features[["ts", *OHLCV_FEATURES_V1]]


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return _wilder(tr, n)


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = _atr(df, n)
    pdi = 100 * _wilder(plus, n) / atr
    mdi = 100 * _wilder(minus, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return _wilder(dx, n)


def _resample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """聚合 k 根 15m K 线，以开盘标记并记录真实收盘时间。 / Aggregate k bars with open label and honest close time."""
    bucket = df["ts"] // (k * BAR_MS) * (k * BAR_MS)
    out = (
        df.groupby(bucket)
        .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .reset_index(names="ts")
    )
    out["close_ts"] = out["ts"] + k * BAR_MS
    return out


def _mtf(df: pd.DataFrame, k: int, tag: str) -> pd.DataFrame:
    h = _resample(df, k)
    logret = np.log(h["close"]).diff()
    return pd.DataFrame(
        {
            "close_ts": h["close_ts"],
            f"f_{tag}_trend": h["close"] / h["close"].ewm(span=20, adjust=False, min_periods=20).mean() - 1,
            f"f_{tag}_vol": logret.rolling(20).std(),
        }
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """每根 15m K 线生成一行以 ts 为键的特征。 / Build one ts-keyed feature row per 15m bar."""
    df = df.sort_values("ts").reset_index(drop=True)
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame({"ts": df["ts"]})

    # --- 价格 / price -------------------------------------------------------
    for n in (1, 2, 4, 8, 16, 32):
        f[f"f_ret_{n}"] = c.pct_change(n, fill_method=None)
    f["f_hl_range"] = (h - lo) / c
    f["f_close_pos_in_range"] = (c - lo) / (h - lo).replace(0, np.nan)
    f["f_dist_high_24h"] = c / h.rolling(96).max() - 1
    f["f_dist_low_24h"] = c / lo.rolling(96).min() - 1

    # --- 趋势 / trend -------------------------------------------------------
    ema = {n: c.ewm(span=n, adjust=False, min_periods=n).mean() for n in (20, 60, 120)}
    for n, e in ema.items():
        f[f"f_ema{n}_rel"] = c / e - 1          # 相对值，不用原始价格 / Relative, never a raw level.
    f["f_ema20_slope"] = ema[20].pct_change(4, fill_method=None)
    f["f_ema60_slope"] = ema[60].pct_change(4, fill_method=None)
    f["f_trend_strength"] = ema[20] / ema[60] - 1
    f["f_adx"] = _adx(df)

    # --- 波动率 / volatility ------------------------------------------------
    logret = np.log(c).diff()
    f["f_atr_pct"] = _atr(df) / c
    for tag, n in (("1h", 4), ("4h", 16), ("24h", 96)):
        f[f"f_rv_{tag}"] = logret.rolling(n).std()
    f["f_bb_width"] = 4 * c.rolling(20).std() / c.rolling(20).mean()

    # --- 成交量 / volume ----------------------------------------------------
    vm, vs = v.rolling(96).mean(), v.rolling(96).std()
    f["f_vol_ratio"] = v / vm
    f["f_vol_z"] = (v - vm) / vs
    f["f_vol_trend"] = v.rolling(4).mean() / vm
    if "taker_buy_volume" in df:
        f["f_buy_sell_ratio"] = df["taker_buy_volume"] / v.replace(0, np.nan)

    # --- 多周期上下文 / multi-timeframe context ------------------------------
    f["close_ts"] = df["ts"] + BAR_MS
    for tag, k in MTF.items():
        f = pd.merge_asof(f, _mtf(df, k, tag), on="close_ts", direction="backward")
    f = f.drop(columns="close_ts")

    return f.replace([np.inf, -np.inf], np.nan)
