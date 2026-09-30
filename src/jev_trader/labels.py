"""First-touch label engine (T1.3-T1.5).

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
BARS_PER_MIN = 15  # 1m bars inside one 15m bar
NO_TOUCH = -1


class Outcome(IntEnum):
    SL_FIRST = 0
    TP_FIRST = 1
    TIMEOUT = 2
    AMBIGUOUS = 3


def level_col(level: float, side: str) -> str:
    """0.0075 -> 'up_75' / 'dn_75' (basis points, so no float keys)."""
    return f"{side}_{round(level * 10_000)}"


# --- reference implementation (T1.4) ---------------------------------------

def first_touch_bruteforce(
    high: np.ndarray, low: np.ndarray, start: int, entry: float,
    levels, horizon_min: int,
) -> tuple[dict, dict]:
    """One entry, one 1m bar at a time, early-exit free. Obviously correct, slow.

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


# --- vectorized path (T1.3) -------------------------------------------------

def build_labels(
    df_15m: pd.DataFrame, df_1m: pd.DataFrame, levels, horizon_bars: int,
    chunk: int = 10_000,
) -> pd.DataFrame:
    """15m bars + 1m path -> first-touch offsets per price level."""
    levels = sorted(levels)
    horizon_min = horizon_bars * BARS_PER_MIN

    ts15 = df_15m["ts"].to_numpy()
    ts1 = df_1m["ts"].to_numpy()
    high = df_1m["high"].to_numpy(float)
    low = df_1m["low"].to_numpy(float)
    close = df_1m["close"].to_numpy(float)

    if not np.all(np.diff(ts1) == MIN_MS):
        raise ValueError("1m series has gaps; the window arithmetic assumes it is contiguous")

    # Entry opens one 15m bar after the signal bar, i.e. at that bar's close time.
    entry_ts = ts15 + BAR_MS
    start = np.searchsorted(ts1, entry_ts)

    # Drop the right edge: no full forward horizon means no label, not "no touch".
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
        wh, wl = win_hi[s], win_lo[s]          # fancy-index copies the chunk
        for lvl in levels:
            for side, w, hit in (
                ("up", wh, wh >= e * (1 + lvl)),
                ("dn", wl, wl <= e * (1 - lvl)),
            ):
                # argmax gives 0 both for "first bar" and "never"; mask with any().
                first = np.where(hit.any(1), hit.argmax(1), NO_TOUCH)
                out[level_col(lvl, side)][a:b] = first

    df = pd.DataFrame(out)
    df.attrs["horizon_min"] = horizon_min
    df.attrs["levels"] = levels
    return df


# --- derivation (T1.5) ------------------------------------------------------

def outcomes(
    labels: pd.DataFrame, side: str, tp: float, sl: float, horizon_bars: int,
) -> np.ndarray:
    """Derive one grid cell from the stored offsets. No rescan, any horizon.

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

    # A touch at or beyond the requested horizon did not happen within it.
    tp_hit = (tp_t >= 0) & (tp_t < n)
    sl_hit = (sl_t >= 0) & (sl_t < n)

    res = np.full(len(labels), Outcome.TIMEOUT, dtype=np.int8)
    res[tp_hit & ~sl_hit] = Outcome.TP_FIRST
    res[sl_hit & ~tp_hit] = Outcome.SL_FIRST
    both = tp_hit & sl_hit
    res[both & (tp_t < sl_t)] = Outcome.TP_FIRST
    res[both & (sl_t < tp_t)] = Outcome.SL_FIRST
    res[both & (tp_t == sl_t)] = Outcome.AMBIGUOUS  # same 1m candle: unorderable
    return res


def summary(labels: pd.DataFrame, grid: dict) -> pd.DataFrame:
    """Per-cell base / timeout / ambiguity rates (T1.7 input)."""
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
