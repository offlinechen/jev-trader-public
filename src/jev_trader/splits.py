"""Horizon-aware purged walk-forward splits (T2.1).

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
    train: np.ndarray       # positional indices into the bar array
    test: np.ndarray
    test_start: int         # ms
    test_end: int
    purged: int             # train bars dropped by the purge


def label_window(ts: np.ndarray, horizon_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """[first, last] timestamp each bar's forward label depends on.

    Entry is at open[t+1] = ts[t] + BAR, and the barrier scan runs for
    `horizon_bars` bars from there.
    """
    start = ts + BAR_MS
    return start, start + horizon_bars * BAR_MS


def resolved_before(ts: np.ndarray, horizon_bars: int, start: int) -> np.ndarray:
    """Whether a bar's forward label is complete before ``start``.

    Entry is one bar after the signal, so the label becomes available after
    ``horizon_bars + 1`` bar intervals.  All causal OOS consumers use this
    same boundary helper.
    """
    return np.asarray(ts) + (horizon_bars + 1) * BAR_MS <= start


def walk_forward(
    ts: np.ndarray, horizon_bars: int, train_months: int = 6, test_months: int = 1,
) -> list[Fold]:
    """Rolling train/test folds, purged by label overlap."""
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

        # Purge: any training bar whose label window touches the test window.
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
