"""按预测时长清除重叠样本的滚动切分（T2.1）。 / Horizon-aware purged walk-forward splits.

相邻信号共享大部分未来路径，普通交叉验证会泄露测试期信息。
只要训练标签窗口与测试窗口相交，就删去该训练样本；规则对测试区间两端均有效。

Barrier labels overlap: bar t and bar t+1 share 15 of 16 bars of forward path.
Naive cross-validation therefore leaks the test period into training and every
downstream metric is inflated.

The purge rule here is the direct statement of the requirement rather than a
fixed gap: **a training bar is dropped whenever its forward label window
intersects the test window.** That handles both edges of the test block
symmetrically and stays correct if the training window ever moves to the right
of the test window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BAR_MS = 900_000


@dataclass(frozen=True)
class Fold:
    fold: int
    train: np.ndarray       # K 线数组的位置索引 / Positional indices into the bar array.
    test: np.ndarray
    test_start: int         # 毫秒 / Milliseconds.
    test_end: int
    purged: int             # 清除的训练 K 线数 / Training bars removed by the purge.


def label_window(ts: np.ndarray, horizon_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """返回标签依赖的未来时间区间。 / Return the forward time interval each label depends on.

    入场在 t+1 开盘，此后扫描 horizon_bars 根 K 线。

    Entry is at open[t+1] = ts[t] + BAR, and the barrier scan runs for
    `horizon_bars` bars from there.
    """
    start = ts + BAR_MS
    return start, start + horizon_bars * BAR_MS


def resolved_before(ts: np.ndarray, horizon_bars: int, start: int) -> np.ndarray:
    """判断未来标签是否在 start 前完整可用。 / Whether the forward label is complete before start.

    入场比信号晚一根，因此须经过 horizon_bars + 1 个间隔；所有因果 OOS 路径共用此边界。

    Entry is one bar after the signal, so the label becomes available after
    ``horizon_bars + 1`` bar intervals.  All causal OOS consumers use this
    same boundary helper.
    """
    return np.asarray(ts) + (horizon_bars + 1) * BAR_MS <= start


def walk_forward(
    ts: np.ndarray, horizon_bars: int, train_months: int = 6, test_months: int = 1,
) -> list[Fold]:
    """按标签重叠清除样本的滚动训练/测试折。 / Rolling train/test folds purged by label overlap."""
    ts = np.asarray(ts)
    lab_start, lab_end = label_window(ts, horizon_bars)
    period = pd.to_datetime(ts, unit="ms").to_period("M")
    months = period.unique().sort_values()

    folds = []
    for i in range(train_months, len(months) - test_months + 1):
        tr_months = months[i - train_months:i]
        te_months = months[i:i + test_months]

        in_train = period.isin(tr_months)
        in_test = period.isin(te_months)
        if not in_test.any():
            continue

        te_ts = ts[in_test]
        t0, t1 = int(te_ts.min()), int(te_ts.max()) + horizon_bars * BAR_MS

        # 清除标签窗口触及测试窗口的训练 K 线。 / Purge training bars whose label windows touch test.
        overlaps = (lab_end >= t0) & (lab_start <= t1)
        keep = in_train & ~overlaps

        folds.append(Fold(
            fold=len(folds),
            train=np.flatnonzero(keep),
            test=np.flatnonzero(in_test),
            test_start=t0, test_end=t1,
            purged=int((in_train & overlaps).sum()),
        ))
    return folds
