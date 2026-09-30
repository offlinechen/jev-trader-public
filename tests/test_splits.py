"""T2.1：清除规则须按标签时长，而非固定间隔。 / Horizon-aware purge, not a fixed gap.

训练标签窗口不得与测试窗口相交，否则折外预测也会泄漏。

The assertion is the requirement stated directly: no training bar's forward
label window may intersect the test window. If that ever fails, every OOF
prediction downstream is contaminated and every baseline number is inflated.
"""

from __future__ import annotations

import numpy as np
import pytest

from jev_trader.splits import BAR_MS, label_window, walk_forward

HORIZON = 16


def bars(months: int = 30) -> np.ndarray:
    n = months * 30 * 96  # 每天 96 根 15m / Ninety-six 15m bars per day.
    return (1_640_995_200_000 // BAR_MS) * BAR_MS + np.arange(n) * BAR_MS


def test_no_train_label_window_touches_test():
    ts = bars()
    lab_start, lab_end = label_window(ts, HORIZON)
    folds = walk_forward(ts, HORIZON)
    assert folds, "no folds produced"
    for f in folds:
        overlap = (lab_end[f.train] >= f.test_start) & (lab_start[f.train] <= f.test_end)
        assert not overlap.any(), (
            f"fold {f.fold}: {overlap.sum()} training bars leak into the test window"
        )


def test_purge_actually_removes_bars():
    """从不移除样本的清除规则无效。 / A purge that removes nothing is not a purge."""
    folds = walk_forward(bars(), HORIZON)
    assert all(f.purged > 0 for f in folds)
    # 恰好清除窗口触及测试块的 horizon+1 根。 / Exactly horizon+1 bars touch test.
    assert {f.purged for f in folds} == {HORIZON + 1}


def test_longer_horizon_purges_more():
    ts = bars()
    short = walk_forward(ts, 4)[0].purged
    long_ = walk_forward(ts, 64)[0].purged
    assert long_ > short, "purge width must follow the horizon"


def test_train_and_test_are_disjoint_and_ordered():
    for f in walk_forward(bars(), HORIZON):
        assert not np.intersect1d(f.train, f.test).size
        assert f.train.max() < f.test.min()


def test_folds_tile_the_period_without_overlap():
    folds = walk_forward(bars(), HORIZON)
    for a, b in zip(folds, folds[1:]):
        assert not np.intersect1d(a.test, b.test).size
        assert b.test.min() > a.test.min()


def test_insufficient_history_yields_no_folds():
    assert walk_forward(bars(months=3), HORIZON, train_months=6) == []
