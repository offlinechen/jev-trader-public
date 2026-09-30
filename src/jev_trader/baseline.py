"""基线模型（T2.2-T2.4）与滚动运行器（T2.6）。 / Baselines and walk-forward runner.

所有模型均预测 TP_FIRST、SL_FIRST、TIMEOUT 三类；不能把超时算作亏损。
用一个模型覆盖 40 个网格单元，使样本与共享结构得到充分利用。
基线的折外预测会持久化，供 Jev 增量价值检验使用。

Three classes throughout -- TP_FIRST / SL_FIRST / TIMEOUT -- for every model.
Collapsing timeout into loss would reintroduce the labelling distortion the
first-touch layer exists to remove, and in the wide-TP cells timeout is the
*majority* outcome.

One model covers all 40 cells: tp, sl, rr and side ride along as features on a
stacked dataset. Forty separate models would fit forty times less data each and
could not share what the grid has in common.

These baselines define the hurdle for Jev, and their out-of-fold predictions are
a first-class deliverable -- Phase 3's incremental-value test consumes exactly
these rows rather than retraining anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .labels import Outcome, outcomes
from .splits import Fold, walk_forward

N_CLASS = 3
AMBIGUOUS = int(Outcome.AMBIGUOUS)
MODELS = ("climatology", "logit", "lgbm")


def cells(grid: dict) -> list[tuple[str, float, float]]:
    return [(side, tp, sl)
            for side in ("long", "short")
            for tp in grid["tp"]
            for sl in grid["sl"]]


def build_dataset(features: pd.DataFrame, labels: pd.DataFrame, grid: dict):
    """返回时间、特征矩阵、标签矩阵、特征名和单元。 / Return timestamps, F, Y, names, and cells."""
    df = labels[["ts"]].merge(features, on="ts", how="inner")
    fcols = [c for c in df.columns if c.startswith("f_")]

    ok = df[fcols].notna().all(1).to_numpy()          # 预热期 NaN，约 320 根 / Warmup NaNs, ~320 bars.
    df = df.loc[ok].reset_index(drop=True)
    lab = labels.loc[labels["ts"].isin(df["ts"])].reset_index(drop=True)
    lab.attrs["horizon_min"] = grid["horizon_bars"] * 15

    cs = cells(grid)
    Y = np.empty((len(lab), len(cs)), dtype=np.int8)
    for j, (side, tp, sl) in enumerate(cs):
        Y[:, j] = outcomes(lab, side, tp, sl, grid["horizon_bars"])

    return (df["ts"].to_numpy(), df[fcols].to_numpy(np.float32), Y, fcols, cs)


def stack(F: np.ndarray, idx: np.ndarray, cs) -> np.ndarray:
    """将 K 线×特征堆成单元优先的训练矩阵。 / Stack bar features and cell metadata, cell-major."""
    n = len(idx)
    base = np.tile(F[idx], (len(cs), 1))
    meta = np.empty((len(cs) * n, 4), dtype=np.float32)
    for j, (side, tp, sl) in enumerate(cs):
        meta[j * n:(j + 1) * n] = (tp, sl, tp / sl, 1.0 if side == "long" else 0.0)
    return np.hstack([base, meta])


def _clim(y: np.ndarray, cell_of: np.ndarray, n_cells: int) -> np.ndarray:
    """每单元经验先验，是真正的零假设而非随机猜测。 / Per-cell empirical prior, the real null."""
    prior = np.full((n_cells, N_CLASS), 1.0 / N_CLASS)
    for j in range(n_cells):
        m = cell_of == j
        if m.any():
            prior[j] = np.bincount(y[m], minlength=N_CLASS) / m.sum()
    return prior


def run_fold(fold: Fold, ts, F, Y, cs, out_dir: Path, horizon_bars: int, seed: int = 0):
    n_cells = len(cs)

    def prep(idx):
        X = stack(F, idx, cs)
        y = Y[idx].T.ravel()                       # 单元优先，匹配 stack() / Cell-major, matching stack().
        cell_of = np.repeat(np.arange(n_cells), len(idx))
        ts_of = np.tile(ts[idx], n_cells)
        keep = y != AMBIGUOUS                      # 歧义不靠假设消解 / Never resolve ambiguity by assumption.
        return X[keep], y[keep], cell_of[keep], ts_of[keep]

    Xtr, ytr, ctr, _ = prep(fold.train)
    Xte, yte, cte, tste = prep(fold.test)

    preds = {"climatology": _clim(ytr, ctr, n_cells)[cte]}

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    logit.fit(Xtr, ytr)
    preds["logit"] = logit.predict_proba(Xte)

    lgbm = LGBMClassifier(
        objective="multiclass", num_class=N_CLASS, n_estimators=200,
        learning_rate=0.05, num_leaves=31, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(Xtr, ytr)
    preds["lgbm"] = lgbm.predict_proba(Xte)

    side = np.array([c[0] for c in cs])[cte]
    tp = np.array([c[1] for c in cs], dtype=np.float32)[cte]
    sl = np.array([c[2] for c in cs], dtype=np.float32)[cte]

    frames = []
    for name, p in preds.items():
        frames.append(pd.DataFrame({
            "ts": tste, "side": side, "tp": tp, "sl": sl,
            "horizon": np.int16(horizon_bars), "fold": np.int16(fold.fold),
            "model": name, "outcome": yte.astype(np.int8),
            "p_sl_first": p[:, 0].astype(np.float32),
            "p_tp_first": p[:, 1].astype(np.float32),
            "p_timeout": p[:, 2].astype(np.float32),
        }))
    oof = pd.concat(frames, ignore_index=True)
    oof["side"] = oof["side"].astype("category")
    oof["model"] = oof["model"].astype("category")

    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out_dir / f"fold_{fold.fold:03d}.parquet", index=False)
    return len(oof), len(ytr), len(yte)


def run(features, labels, grid, data_dir, train_months=6, test_months=1, max_folds=None):
    ts, F, Y, fcols, cs = build_dataset(features, labels, grid)
    folds = walk_forward(ts, grid["horizon_bars"], train_months, test_months)
    if max_folds:
        folds = folds[:max_folds]

    out_dir = Path(data_dir) / "oof"
    for f in out_dir.glob("fold_*.parquet"):
        f.unlink()

    print(f"{len(ts):,} bars x {len(cs)} cells | {len(fcols)} features | {len(folds)} folds")
    for fold in folds:
        n_oof, n_tr, n_te = run_fold(fold, ts, F, Y, cs, out_dir, grid["horizon_bars"])
        print(
            f"  fold {fold.fold:>3}  train {n_tr:>9,}  test {n_te:>8,}  "
            f"purged {fold.purged:>3} bars  -> {n_oof:,} oof rows"
        )
    return out_dir
