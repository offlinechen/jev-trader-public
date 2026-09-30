"""T2.7：重新读取并评分 OOF 数据。 / Reload and score the OOF dataset.

直接消费持久化预测，而非内存中的模型；若无法复现基线结果，后续增量检验便不可信。

Deliberately consumes `data/oof/` rather than model objects: if the persisted
predictions cannot reproduce the headline numbers, Phase 3's incremental test
cannot trust them either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import CLASS_NAMES, block_bootstrap_ci, evaluate, multiclass_brier, reliability

PROB_COLS = ["p_sl_first", "p_tp_first", "p_timeout"]


def load_oof(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "oof")


def _yp(df: pd.DataFrame):
    return df["outcome"].to_numpy(), df[PROB_COLS].to_numpy(np.float64)


def pooled(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in oof.groupby("model", observed=True):
        y, p = _yp(g)
        rows.append({"model": model, **evaluate(y, p)})
    return pd.DataFrame(rows).set_index("model")


def by_cell(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, side, tp, sl), g in oof.groupby(
        ["model", "side", "tp", "sl"], observed=True
    ):
        y, p = _yp(g)
        r = evaluate(y, p)
        rows.append({
            "model": model, "side": side, "tp": tp, "sl": sl,
            "n": r["n"], "brier_mc": r["brier_mc"],
            "auc_tp_first": r["auc_tp_first"], "auc_tp_vs_sl": r["auc_tp_vs_sl"],
            "base_timeout": r["base_timeout"],
        })
    return pd.DataFrame(rows)


def by_fold(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, fold), g in oof.groupby(["model", "fold"], observed=True):
        y, p = _yp(g)
        r = evaluate(y, p)
        rows.append({"model": model, "fold": fold, "n": r["n"],
                     "brier_mc": r["brier_mc"], "auc_tp_first": r["auc_tp_first"]})
    return pd.DataFrame(rows)


def ci_for(oof: pd.DataFrame, model: str) -> dict:
    g = oof[oof["model"] == model]
    y, p = _yp(g)
    ts = g["ts"].to_numpy()
    lo, hi = block_bootstrap_ci(ts, y, p, multiclass_brier)
    return {"brier_lo": lo, "brier_hi": hi}


def _cell_auc_tp_vs_sl(y: np.ndarray, score: np.ndarray) -> float:
    pos = y == 1
    if not (0 < pos.sum() < len(y)):
        return np.nan
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(pos.astype(int), score)


def within_cell_auc_ci(
    oof: pd.DataFrame, model: str, n_boot: int = 200, seed: int = 0,
) -> tuple[float, float, float]:
    """单元内平均 auc_tp_vs_sl 的整日聚类置信区间。 / Day-clustered CI for within-cell AUC.

同一根 K 线的各单元共享价格路径，相邻 K 线的未来窗口也重叠；
将单元行视为独立会虚假缩窄置信区间。

    Resampling whole DAYS is what makes this honest: the 40 cells of a single
    bar are 40 views of one price path, and overlapping 4h horizons make
    consecutive bars dependent too. Treating cell-rows as independent would
    shrink the interval by roughly the square root of 40 and manufacture
    significance that is not there.
    """
    g = oof[(oof["model"] == model) & oof["outcome"].isin([0, 1])]
    day = (g["ts"].to_numpy() // 86_400_000)
    cell = (g["side"].astype(str) + "_" + g["tp"].astype(str) + "_" + g["sl"].astype(str)).to_numpy()
    y = g["outcome"].to_numpy()
    p_tp, p_sl = g["p_tp_first"].to_numpy(), g["p_sl_first"].to_numpy()
    score = p_tp / np.maximum(p_tp + p_sl, 1e-12)

    cells_u = np.unique(cell)
    days_u = np.unique(day)
    idx_by_day = {d: np.flatnonzero(day == d) for d in days_u}

    point = np.nanmean([
        _cell_auc_tp_vs_sl(y[cell == c], score[cell == c]) for c in cells_u
    ])

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_day[d] for d in rng.choice(days_u, len(days_u), True)])
        yc, sc, cc = y[pick], score[pick], cell[pick]
        vals.append(np.nanmean([
            _cell_auc_tp_vs_sl(yc[cc == c], sc[cc == c]) for c in cells_u
        ]))
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def reliability_table(oof: pd.DataFrame, model: str, cls: int = 1) -> pd.DataFrame:
    g = oof[oof["model"] == model]
    y, p = _yp(g)
    return reliability((y == cls).astype(int), p[:, cls])


def within_cell(oof: pd.DataFrame) -> pd.DataFrame:
    """逐单元计算后取平均，避免混合单元 AUC 误导。 / Average per-cell metrics, the honest view.

不同网格单元基础概率差异很大，混合 AUC 会把网格几何结构误认为预测能力。

    **Pooled AUC across cells is not a skill measure.** The 40 cells have base
    rates from 0.40 down to 0.02, so merely knowing which cell a row belongs to
    ranks TP-first well: pooled AUC for climatology, which has no features at
    all, lands near 0.78. That is grid geometry, not forecasting. Every
    headline ranking number is therefore computed per cell and then averaged.
    """
    return (
        by_cell(oof)
        .groupby("model", observed=True)[["auc_tp_first", "auc_tp_vs_sl", "brier_mc"]]
        .mean()
    )


def summary(data_dir: str | Path) -> dict:
    oof = load_oof(data_dir)
    pool, wc = pooled(oof), within_cell(oof)

    # 方向门槛固定为 LightGBM；TP 是否发生多半只是波动率预测。 / Fix the directional hurdle to LightGBM; TP occurrence mostly predicts volatility.
    # auc_tp_vs_sl 检验 Jev 是否增加押中方向的信息。 / Test whether Jev adds information about the winning direction.
    return {
        "oof": oof,
        "pooled": pool,              # 仅诊断，见 within_cell() / Diagnostic only; see within_cell().
        "within_cell": wc,
        "by_cell": by_cell(oof),
        "by_fold": by_fold(oof),
        "A_base_direction": float(wc.loc["lgbm", "auc_tp_vs_sl"]),
        "A_vol": float(wc.loc["lgbm", "auc_tp_first"]),
        "A_vol_logit": float(wc.loc["logit", "auc_tp_first"]),
        "B_clim": float(pool.loc["climatology", "brier_mc"]),
        "B_lgbm": float(pool.loc["lgbm", "brier_mc"]),
        "B_logit": float(pool.loc["logit", "brier_mc"]),
        "classes": CLASS_NAMES,
    }
