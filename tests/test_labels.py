"""T1.4/T1.5：快速路径只有与参考实现一致才可信。
/ The fast path is trusted only where it matches the reference.

逐分钟暴力扫描定义正确性；向量化只是优化，不一致就是缺陷。

`first_touch_bruteforce` is the definition of correct: one entry, one 1m bar at
a time. `build_labels` is an optimisation, and an optimisation that disagrees
with its reference is a bug no matter how fast it is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jev_trader.labels import (
    BAR_MS, MIN_MS, NO_TOUCH, Outcome, build_labels, first_touch_bruteforce,
    level_col, outcomes,
)

LEVELS = [0.005, 0.0075, 0.010, 0.015, 0.020, 0.030]
HORIZON = 16
DATA = Path(__file__).resolve().parents[1] / "data" / "ohlcv"


def synthetic_1m(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.0006, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.0008, n)) * close
    high = np.maximum.reduce([close + wick * rng.random(n), open_, close])
    low = np.minimum.reduce([close - wick * rng.random(n), open_, close])
    anchor = (1_600_000_000_000 // BAR_MS) * BAR_MS
    return pd.DataFrame({
        "ts": anchor + np.arange(n) * MIN_MS,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(1, 0.5, n),
    })


def to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    b = df_1m["ts"] // BAR_MS * BAR_MS
    return (
        df_1m.groupby(b)
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        .reset_index(names="ts")
    )


def _real():
    f15, f1 = DATA / "BTCUSDT_15m.parquet", DATA / "BTCUSDT_1m.parquet"
    if not (f15.exists() and f1.exists()):
        return None
    return pd.read_parquet(f15), pd.read_parquet(f1)


# --- T1.4：向量化等于暴力扫描 / vectorized equals brute force -----------------

def _compare(df_15m, df_1m, k, seed):
    labels = build_labels(df_15m, df_1m, LEVELS, HORIZON)
    high = df_1m["high"].to_numpy(float)
    low = df_1m["low"].to_numpy(float)
    ts1 = df_1m["ts"].to_numpy()
    horizon_min = HORIZON * 15

    rng = np.random.default_rng(seed)
    for i in rng.choice(len(labels), min(k, len(labels)), replace=False):
        row = labels.iloc[i]
        start = int(np.searchsorted(ts1, row["entry_ts"]))
        assert ts1[start] == row["entry_ts"]
        up, dn = first_touch_bruteforce(
            high, low, start, row["entry_price"], LEVELS, horizon_min
        )
        for lvl in LEVELS:
            assert row[level_col(lvl, "up")] == up[lvl], (
                f"up {lvl} at ts={row['ts']}: fast={row[level_col(lvl,'up')]} ref={up[lvl]}"
            )
            assert row[level_col(lvl, "dn")] == dn[lvl], (
                f"dn {lvl} at ts={row['ts']}: fast={row[level_col(lvl,'dn')]} ref={dn[lvl]}"
            )


def test_vectorized_matches_bruteforce_synthetic():
    df_1m = synthetic_1m()
    _compare(to_15m(df_1m), df_1m, k=300, seed=1)


@pytest.mark.skipif(_real() is None, reason="real OHLCV not downloaded")
def test_vectorized_matches_bruteforce_real():
    df_15m, df_1m = _real()
    _compare(df_15m, df_1m, k=1000, seed=2)


# --- T1.3：右边界、入场价、连续性 / right edge, entry, continuity ------------

def test_incomplete_horizon_is_dropped_not_labelled():
    df_1m = synthetic_1m()
    df_15m = to_15m(df_1m)
    labels = build_labels(df_15m, df_1m, LEVELS, HORIZON)
    # 每个有效标签都有完整的 1m 未来窗口。 / Every labelled bar has a full 1m forward window.
    last_needed = labels["entry_ts"] + HORIZON * BAR_MS
    assert (last_needed <= df_1m["ts"].iloc[-1] + MIN_MS).all()
    # 被删除的尾部正是缺少未来窗口的 K 线。 / The dropped tail lacks that forward window.
    assert labels["ts"].iloc[-1] < df_15m["ts"].iloc[-1]


def test_entry_is_next_bar_open():
    df_1m = synthetic_1m()
    df_15m = to_15m(df_1m)
    labels = build_labels(df_15m, df_1m, LEVELS, HORIZON).head(200)
    nxt = df_15m.set_index("ts")["open"]
    for _, r in labels.iterrows():
        assert r["entry_price"] == pytest.approx(nxt.loc[r["ts"] + BAR_MS])


def test_gappy_1m_is_rejected():
    df_1m = synthetic_1m()
    df_15m = to_15m(df_1m)
    with pytest.raises(ValueError, match="gaps"):
        build_labels(df_15m, df_1m.drop(index=5000), LEVELS, HORIZON)


# --- T1.5：结果推导 / outcome derivation ------------------------------------

def _labels_from(up_t, dn_t, horizon_min=HORIZON * 15):
    df = pd.DataFrame({level_col(0.01, "up"): [up_t], level_col(0.01, "dn"): [dn_t]})
    df.attrs["horizon_min"] = horizon_min
    return df


@pytest.mark.parametrize("up_t,dn_t,expected", [
    (10, 50, Outcome.TP_FIRST),
    (50, 10, Outcome.SL_FIRST),
    (10, NO_TOUCH, Outcome.TP_FIRST),
    (NO_TOUCH, 10, Outcome.SL_FIRST),
    (NO_TOUCH, NO_TOUCH, Outcome.TIMEOUT),
    (33, 33, Outcome.AMBIGUOUS),          # 同根 1m 不猜顺序 / Same 1m; never assume order.
])
def test_outcome_long(up_t, dn_t, expected):
    assert outcomes(_labels_from(up_t, dn_t), "long", 0.01, 0.01, HORIZON)[0] == expected


@pytest.mark.parametrize("up_t,dn_t,expected", [
    (10, 50, Outcome.SL_FIRST),           # 多头镜像 / Mirror of the long case.
    (50, 10, Outcome.TP_FIRST),
    (33, 33, Outcome.AMBIGUOUS),
])
def test_outcome_short(up_t, dn_t, expected):
    assert outcomes(_labels_from(up_t, dn_t), "short", 0.01, 0.01, HORIZON)[0] == expected


def test_shorter_horizon_truncates_without_relabelling():
    lab = _labels_from(up_t=100, dn_t=200)
    assert outcomes(lab, "long", 0.01, 0.01, 16)[0] == Outcome.TP_FIRST   # 分钟 / Minutes: 240.
    assert outcomes(lab, "long", 0.01, 0.01, 8)[0] == Outcome.TP_FIRST    # 分钟 / Minutes: 120.
    assert outcomes(lab, "long", 0.01, 0.01, 4)[0] == Outcome.TIMEOUT     # 分钟 / Minutes: 60.


def test_horizon_beyond_labelled_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        outcomes(_labels_from(10, 20), "long", 0.01, 0.01, HORIZON + 1)


def test_monotone_in_level():
    """近障碍不可能比远障碍更晚触及。 / A nearer barrier cannot be touched later than a farther one."""
    df_1m = synthetic_1m()
    lab = build_labels(to_15m(df_1m), df_1m, LEVELS, HORIZON)
    for side in ("up", "dn"):
        for near, far in zip(LEVELS, LEVELS[1:]):
            n = lab[level_col(near, side)].to_numpy()
            f = lab[level_col(far, side)].to_numpy()
            both = (n >= 0) & (f >= 0)
            assert (n[both] <= f[both]).all(), f"{side} {near} touched after {far}"
            assert not ((n == NO_TOUCH) & (f >= 0)).any(), f"{side}: far hit, near not"


# --- 永久不变量回归测试 / permanent invariant regression tests ---------------
# 这两个性质来自 T1.7 报告，可低成本捕获符号或索引错误，因此保留为回归测试。
# These T1.7 properties cheaply catch sign/index errors and remain regression tests.

GRID_TP = [0.005, 0.010, 0.015, 0.020, 0.030]
GRID_SL = [0.005, 0.0075, 0.010, 0.015]


def _rates(lab, side, tp, sl):
    r = outcomes(lab, side, tp, sl, HORIZON)
    resolved = r != Outcome.AMBIGUOUS
    n = resolved.sum()
    return ((r == Outcome.TP_FIRST).sum() / n, (r == Outcome.SL_FIRST).sum() / n)


@pytest.mark.skipif(_real() is None, reason="real OHLCV not downloaded")
def test_long_short_mirror_symmetry_is_exact():
    """多头 (tp=a,sl=b) 与空头 (tp=b,sl=a) 是镜像事件，必须精确一致。

    long(tp=a, sl=b) and short(tp=b, sl=a) are the same event, mirrored.

    The two run through independent branches of outcomes(); they must agree
    exactly, not approximately.
    """
    df_15m, df_1m = _real()
    lab = build_labels(df_15m, df_1m, LEVELS, HORIZON)
    for a in GRID_TP:
        for b in GRID_SL:
            long_tp, long_sl = _rates(lab, "long", a, b)
            short_tp, short_sl = _rates(lab, "short", b, a)
            assert long_tp == pytest.approx(short_sl, abs=1e-12), f"tp {a}/{b}"
            assert long_sl == pytest.approx(short_tp, abs=1e-12), f"sl {a}/{b}"


@pytest.mark.skipif(_real() is None, reason="real OHLCV not downloaded")
def test_empirical_monotonicity_holds():
    """Jev 矩阵的 FR-5 单调约束也是数据自身的性质。

    The FR-5 constraints imposed on Jev's matrix are properties of the data.

    p_tp falls as TP widens (fixed SL) and rises as SL widens (fixed TP).
    """
    df_15m, df_1m = _real()
    lab = build_labels(df_15m, df_1m, LEVELS, HORIZON)
    for side in ("long", "short"):
        for sl in GRID_SL:
            series = [_rates(lab, side, tp, sl)[0] for tp in GRID_TP]
            assert all(x >= y for x, y in zip(series, series[1:])), \
                f"{side} sl={sl}: p_tp not non-increasing in tp: {series}"
        for tp in GRID_TP:
            series = [_rates(lab, side, tp, sl)[0] for sl in GRID_SL]
            assert all(x <= y for x, y in zip(series, series[1:])), \
                f"{side} tp={tp}: p_tp not non-decreasing in sl: {series}"
