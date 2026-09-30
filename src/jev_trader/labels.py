"""首触标签引擎（T1.3-T1.5）。 / First-touch label engine.

标签按价格距离而非 TP/SL 组合存储首次触及分钟，因此 1m 路径只需扫描一次，
更改网格或缩短时长都无需重新标注。信号在 t 收盘产生，t+1 开盘入场；
同一分钟同时触及 TP/SL 记为歧义，未来窗口不完整的样本直接剔除。

The label layer is **price-level based, not TP/SL-pair based**. For each 15m
entry bar we store, per price offset, the minute at which that offset was first
touched. Every TP/SL matrix cell and every horizon <= the stored one is then a
comparison between two of those numbers -- so the 1m path is scanned once, not
40 times, and the dataset is strategy-independent: changing the grid or the
horizon requires no relabelling.

Conventions, all four load-bearing:

* **Entry is `open[t+1]`**, the bar after the signal bar, because that is where
  Freqtrade actually fills. Labelling from `close[t]` would build a systematic
  gap between research and execution.
* **Offsets are minutes from entry**, -1 for "not touched within the stored
  horizon". Minute resolution is what makes any shorter horizon derivable.
* **Same-minute ties are AMBIGUOUS**, never resolved by assumption. When both
  barriers fall in one 1m candle the OHLC cannot order them; callers exclude
  those rows rather than guess.
* **Bars without a full forward horizon are dropped**, never encoded as "no
  touch". An absent future is not evidence of a non-event.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BAR_MS = 900_000
MIN_MS = 60_000
BARS_PER_MIN = 15  # 每根 15m 内有 15 根 1m / Fifteen 1m bars per 15m bar.
NO_TOUCH = -1


class Outcome(IntEnum):
    SL_FIRST = 0
    TP_FIRST = 1
    TIMEOUT = 2
    AMBIGUOUS = 3


def level_col(level: float, side: str) -> str:
    """将 0.0075 转为基点键，避免浮点键。 / Turn 0.0075 into a basis-point key, not a float key."""
    return f"{side}_{round(level * 10_000)}"


# --- 参考实现 / reference implementation (T1.4) -----------------------------

def first_touch_bruteforce(
    high: np.ndarray, low: np.ndarray, start: int, entry: float,
    levels, horizon_min: int,
) -> tuple[dict, dict]:
    """逐笔逐分钟扫描，不提前退出；慢但易于核对。 / Scan one entry and minute at a time; slow but clear.

    这是向量化路径必须精确匹配的参考实现（tests/test_labels.py）。

    This is the reference. The vectorized path is only trusted where it matches
    this exactly (tests/test_labels.py).
    """
    up = {lvl: NO_TOUCH for lvl in levels}
    dn = {lvl: NO_TOUCH for lvl in levels}
    for k in range(horizon_min):
        h, lo = high[start + k], low[start + k]
        for lvl in levels:
            if up[lvl] == NO_TOUCH and h >= entry * (1 + lvl):
                up[lvl] = k
            if dn[lvl] == NO_TOUCH and lo <= entry * (1 - lvl):
                dn[lvl] = k
    return up, dn


# --- 向量化路径 / vectorized path (T1.3) ------------------------------------

def build_labels(
    df_15m: pd.DataFrame, df_1m: pd.DataFrame, levels, horizon_bars: int,
    chunk: int = 10_000,
) -> pd.DataFrame:
    """由 15m K 线和 1m 路径计算每个价位的首触偏移。 / Compute first-touch offsets from 15m/1m bars."""
    levels = sorted(levels)
    horizon_min = horizon_bars * BARS_PER_MIN

    ts15 = df_15m["ts"].to_numpy()
    ts1 = df_1m["ts"].to_numpy()
    high = df_1m["high"].to_numpy(float)
    low = df_1m["low"].to_numpy(float)
    close = df_1m["close"].to_numpy(float)

    if not np.all(np.diff(ts1) == MIN_MS):
        raise ValueError("1m series has gaps; the window arithmetic assumes it is contiguous")

    # 信号后一根 15m 开盘入场，即信号 K 线收盘时刻。 / Enter at the next 15m open, at signal-bar close.
    entry_ts = ts15 + BAR_MS
    start = np.searchsorted(ts1, entry_ts)

    # 右边界没有完整未来窗口就剔除，不能记为未触及。 / Drop incomplete futures, not "no touch".
    ok = (start < len(ts1)) & (start + horizon_min <= len(ts1))
    ok &= np.take(ts1, np.clip(start, 0, len(ts1) - 1)) == entry_ts
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise ValueError("no 15m bar has a complete forward horizon")

    start = start[idx]
    entry = np.take(df_1m["open"].to_numpy(float), start)

    out = {
        "ts": ts15[idx],
        "entry_ts": entry_ts[idx],
        "entry_price": entry,
        "ret_at_horizon": close[start + horizon_min - 1] / entry - 1,
    }
    for lvl in levels:
        for side in ("up", "dn"):
            out[level_col(lvl, side)] = np.full(idx.size, NO_TOUCH, dtype=np.int16)

    win_hi = sliding_window_view(high, horizon_min)
    win_lo = sliding_window_view(low, horizon_min)

    for a in range(0, idx.size, chunk):
        b = min(a + chunk, idx.size)
        s = start[a:b]
        e = entry[a:b, None]
        wh, wl = win_hi[s], win_lo[s]          # 花式索引会复制分块 / Fancy indexing copies the chunk.
        for lvl in levels:
            for side, w, hit in (
                ("up", wh, wh >= e * (1 + lvl)),
                ("dn", wl, wl <= e * (1 - lvl)),
            ):
                # argmax 对首分钟和从未触及都返回 0，需用 any() 区分。 / Mask argmax's zero with any().
                first = np.where(hit.any(1), hit.argmax(1), NO_TOUCH)
                out[level_col(lvl, side)][a:b] = first

    df = pd.DataFrame(out)
    df.attrs["horizon_min"] = horizon_min
    df.attrs["levels"] = levels
    return df


# --- 结果推导 / derivation (T1.5) ------------------------------------------

def outcomes(
    labels: pd.DataFrame, side: str, tp: float, sl: float, horizon_bars: int,
) -> np.ndarray:
    """由已存首触偏移推导单元结果，无需重扫。 / Derive one cell from stored offsets without rescanning.

    多头上涨止盈、下跌止损；空头相反。允许使用不超过原标签窗口的更短时长。

    A long takes profit on an up move and stops on a down move; a short is the
    mirror. `horizon_bars` may be anything up to the horizon the labels were
    built with -- shorter horizons are a comparison, not a relabelling.
    """
    if side == "long":
        tp_t = labels[level_col(tp, "up")].to_numpy()
        sl_t = labels[level_col(sl, "dn")].to_numpy()
    elif side == "short":
        tp_t = labels[level_col(tp, "dn")].to_numpy()
        sl_t = labels[level_col(sl, "up")].to_numpy()
    else:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    n = horizon_bars * BARS_PER_MIN
    stored = labels.attrs.get("horizon_min")
    if stored is not None and n > stored:
        raise ValueError(f"horizon {n}m exceeds the labelled horizon {stored}m")

    # 达到或晚于指定时长的触及，不算窗口内事件。 / A touch at or after the horizon is outside it.
    tp_hit = (tp_t >= 0) & (tp_t < n)
    sl_hit = (sl_t >= 0) & (sl_t < n)

    res = np.full(len(labels), Outcome.TIMEOUT, dtype=np.int8)
    res[tp_hit & ~sl_hit] = Outcome.TP_FIRST
    res[sl_hit & ~tp_hit] = Outcome.SL_FIRST
    both = tp_hit & sl_hit
    res[both & (tp_t < sl_t)] = Outcome.TP_FIRST
    res[both & (sl_t < tp_t)] = Outcome.SL_FIRST
    res[both & (tp_t == sl_t)] = Outcome.AMBIGUOUS  # 同一 1m 无法判顺序 / Same 1m cannot order touches.
    return res


def summary(labels: pd.DataFrame, grid: dict) -> pd.DataFrame:
    """每单元的基准、超时和歧义率（T1.7 输入）。 / Per-cell base, timeout, and ambiguity rates."""
    rows = []
    for side in ("long", "short"):
        for tp in grid["tp"]:
            for sl in grid["sl"]:
                res = outcomes(labels, side, tp, sl, grid["horizon_bars"])
                n = len(res)
                amb = (res == Outcome.AMBIGUOUS).sum()
                resolved = n - amb
                rows.append({
                    "side": side, "tp": tp, "sl": sl, "rr": round(tp / sl, 2),
                    "n": n,
                    "p_tp": (res == Outcome.TP_FIRST).sum() / max(resolved, 1),
                    "p_sl": (res == Outcome.SL_FIRST).sum() / max(resolved, 1),
                    "p_timeout": (res == Outcome.TIMEOUT).sum() / max(resolved, 1),
                    "ambiguous_pct": 100 * amb / n,
                })
    return pd.DataFrame(rows)
