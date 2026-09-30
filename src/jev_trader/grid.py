"""Shared barrier-cell keys and monotone barrier-surface projection."""

from __future__ import annotations

import numpy as np


def level_code(level: float) -> str:
    return f"{round(float(level) * 10_000):03d}"


def cell_key(side: str, tp: float, sl: float) -> str:
    return f"{side}_tp{level_code(tp)}_sl{level_code(sl)}"


def tp_key(side: str, tp: float, sl: float) -> str:
    return f"p_{cell_key(side, tp, sl)}"


def sl_key(side: str, tp: float, sl: float) -> str:
    return f"p_{side}_sl{level_code(sl)}_tp{level_code(tp)}"


def _pava(values: np.ndarray, *, increasing: bool) -> np.ndarray:
    """Return the closest 1-D isotonic sequence using unit weights."""
    values = np.asarray(values, dtype=float)
    sign = 1.0 if increasing else -1.0
    blocks: list[list[float]] = []
    for value in sign * values:
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            left, right = blocks[-2], blocks[-1]
            weight = left[1] + right[1]
            blocks[-2:] = [[(left[0] * left[1] + right[0] * right[1]) / weight, weight]]
    projected = np.concatenate([
        np.full(int(weight), mean, dtype=float) for mean, weight in blocks
    ])
    return sign * projected


def _project_surface(matrix: np.ndarray, *, rows_increasing: bool, columns_increasing: bool,
                     rounds: int = 3) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    for _ in range(rounds):
        for row in range(result.shape[0]):
            result[row, :] = _pava(result[row, :], increasing=rows_increasing)
        for column in range(result.shape[1]):
            result[:, column] = _pava(result[:, column], increasing=columns_increasing)
    return result


def project_tp_surface(matrix: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Project TP probability: farther SL helps, farther TP hurts."""
    return _project_surface(matrix, rows_increasing=True, columns_increasing=False, rounds=rounds)


def project_sl_surface(matrix: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Project SL probability: farther TP helps, farther SL hurts."""
    return _project_surface(matrix, rows_increasing=False, columns_increasing=True, rounds=rounds)


def project_barrier_surfaces(tp_matrix: np.ndarray, sl_matrix: np.ndarray, rounds: int = 3):
    """Project both surfaces without clipping or silently normalizing them."""
    projected_tp = project_tp_surface(tp_matrix, rounds=rounds)
    projected_sl = project_sl_surface(sl_matrix, rounds=rounds)
    probability_sum = projected_tp + projected_sl
    return projected_tp, projected_sl, 1.0 - probability_sum, probability_sum


def project_monotone(matrix: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Compatibility wrapper; new callers should use named surface functions."""
    return project_sl_surface(matrix) if reverse else project_tp_surface(matrix)
