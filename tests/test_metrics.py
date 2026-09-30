"""Metric guards.

The first test exists because pooled AUC across the barrier grid fooled us
once: climatology, which has no features, scores ~0.78 pooled purely because
the 40 cells have base rates from 0.40 down to 0.02. Ranking rows by "which
cell is this" is grid geometry, not skill.

This test pins that down with synthetic data so nobody restores pooled AUC as
a headline metric later. If it ever fails, the metric layer has drifted back
to measuring the grid instead of the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jev_trader.metrics import ece, evaluate, multiclass_brier


def test_pooled_auc_is_inflated_by_cell_structure():
    rng = np.random.default_rng(0)
    base_rates = [0.40, 0.20, 0.10, 0.02]   # like tp=0.5% .. 3%
    y, p, cell = [], [], []
    for j, r in enumerate(base_rates):
        n = 5000
        y.append(rng.random(n) < r)
        p.append(np.full(n, r))             # constant within cell: zero skill
        cell.append(np.full(n, j))
    y = np.concatenate(y).astype(int)
    p = np.concatenate(p)
    cell = np.concatenate(cell)

    pooled = roc_auc_score(y, p)
    per_cell = [
        roc_auc_score(y[cell == j], p[cell == j])
        if 0 < y[cell == j].sum() < (cell == j).sum() else np.nan
        for j in range(len(base_rates))
    ]
    assert pooled > 0.70, "the trap should be visible"
    assert np.allclose(np.nanmean(per_cell), 0.5, atol=1e-9), (
        "a within-cell constant predictor must score exactly 0.5"
    )


def test_multiclass_brier_bounds():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert multiclass_brier(y, perfect) == 0.0
    worst = np.array([[0, 0, 1.0], [0, 0, 1.0], [1.0, 0, 0]])
    assert multiclass_brier(y, worst) == 2.0
    uniform = np.full((3, 3), 1 / 3)
    assert multiclass_brier(y, uniform) == np.float64(2 / 3)


def test_brier_prefers_the_better_forecast():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 20_000)
    good = np.full((len(y), 3), 0.05)
    good[np.arange(len(y)), y] = 0.90
    bad = np.full((len(y), 3), 1 / 3)
    assert multiclass_brier(y, good) < multiclass_brier(y, bad)


def test_ece_zero_for_calibrated_and_large_for_biased():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 50_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert ece(y, p) < 0.01
    assert ece(y, np.clip(p + 0.25, 0, 1)) > 0.15


def test_evaluate_keeps_three_classes():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 3000)
    p = rng.dirichlet([1, 1, 1], 3000)
    r = evaluate(y, p)
    for name in ("sl_first", "tp_first", "timeout"):
        assert f"auc_{name}" in r and f"base_{name}" in r
    assert abs(sum(r[f"base_{n}"] for n in ("sl_first", "tp_first", "timeout")) - 1) < 1e-9
