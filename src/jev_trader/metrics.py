"""Probability-quality metrics (T2.5).

Everything here is three-class: TP_FIRST / SL_FIRST / TIMEOUT. Nothing collapses
timeout into loss -- with timeout rates of 70-80% in the wide-TP cells that
would recreate exactly the labelling distortion the first-touch layer removed.

Trading PnL is deliberately absent. It belongs downstream of a calibrated
probability, not alongside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

N_CLASS = 3
CLASS_NAMES = ["sl_first", "tp_first", "timeout"]  # index == Outcome value


def multiclass_brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error over the full probability vector (0 = perfect, 2 = worst)."""
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(1).mean())


def ece(y_bin: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins."""
    if len(p) < bins * 2:
        return float("nan")
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y_bin[m].mean())
    return float(err)


def reliability(y_bin: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()),
                         "p_mean": float(p[m].mean()), "observed": float(y_bin[m].mean())})
    return pd.DataFrame(rows)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    """Full three-class report for one (model, slice)."""
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum(1, keepdims=True)
    out = {
        "n": int(len(y)),
        "brier_mc": multiclass_brier(y, p),
        "logloss": float(log_loss(y, p, labels=list(range(N_CLASS)))),
    }
    for k, name in enumerate(CLASS_NAMES):
        y_bin = (y == k).astype(int)
        out[f"auc_{name}"] = (
            float(roc_auc_score(y_bin, p[:, k])) if 0 < y_bin.sum() < len(y_bin) else float("nan")
        )
        out[f"ece_{name}"] = ece(y_bin, p[:, k])
        out[f"base_{name}"] = float(y_bin.mean())
    # The tradeable signal: can the model rank TP-first ahead of SL-first?
    m = np.isin(y, [0, 1])
    if m.sum() > 1 and 0 < (y[m] == 1).sum() < m.sum():
        score = p[m, 1] / (p[m, 1] + p[m, 0])
        out["auc_tp_vs_sl"] = float(roc_auc_score((y[m] == 1).astype(int), score))
    else:
        out["auc_tp_vs_sl"] = float("nan")
    return out


def block_bootstrap_ci(
    ts: np.ndarray, y: np.ndarray, p: np.ndarray, stat, n: int = 200,
    block_ms: int = 86_400_000, seed: int = 0,
) -> tuple[float, float]:
    """95% CI resampling whole days, because overlapping labels are not independent."""
    day = ts // block_ms
    days = np.unique(day)
    by_day = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(days, len(days), replace=True)
        idx = np.concatenate([by_day[d] for d in pick])
        try:
            vals.append(stat(y[idx], p[idx]))
        except ValueError:
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
