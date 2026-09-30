"""校准后 TP/SL 矩阵上的策略政策；只是实验，不是默认交易策略。
/ Experimental policies over the calibrated TP/SL matrix, not a trading default.

固定单元是低选择偏差的参照；argmax、局部稳健、softmax 和多 TP 是对照。
每条腿对应一个网格单元，实际收益按权重加总，不需要近似。
所有政策共用 weights(ev)，参数在评估前固定，不在这里调优。

Hypothesis under test: the probability matrix is a distributional object, and
hard-selecting its maximum cell (40 noisy "lottery tickets", keep the best)
is structurally fragile because of the winner's curse. Policies:

* `FixedCellPolicy`   -- one pre-committed geometry; only the side is chosen.
                         The reference: almost no optimisation over noise.
* `ArgmaxEVPolicy`    -- the current design: the single max-EV cell of 40.
* `LocalRobustPolicy` -- argmax of local-mean(EV) - lambda * local-std(EV) over
                         each cell's 3x3 grid neighbourhood: prefers broad,
                         supported plateaus over isolated spikes.
* `SoftmaxEVPolicy`   -- softmax(EV / tau) over the chosen side's 20 cells: a
                         portfolio of micro-positions, one per cell.
* `MultiTPPolicy`     -- one entry, one shared stop, softmax(EV / tau) across TP
                         levels as partial exits. Executable as a single order set.

Why every policy is realised *exactly*, with no approximation: each leg of a
plan is itself a grid cell. A softmax portfolio leg is an independent position
with its own TP and SL; a multi-TP tranche with a shared, unmoved stop exits at
its TP if touched before the stop, at the stop otherwise, or at the horizon --
which is precisely the first-touch outcome of the cell (TP_j, SL). So realised
policy return is `sum_i w_i * net_i` over the cells' realised returns, and
fees scale with notional the same way.

All policies share one vectorised core, `weights(ev) -> w`: the evaluator
calls it on every bar at once, and `build_plan` calls it on a single bar for
live use. Research and live cannot drift apart.

Parameters (tau, lambda, fixed geometries) are fixed *before* evaluation and
reported side by side as a stability check. None of them is tuned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .splits import resolved_before

SIDES = ("long", "short")
DAY_MS = 86_400_000


# --- 交易计划 / plan ---------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    tp: float
    sl: float
    weight: float


@dataclass(frozen=True)
class TradePlan:
    side: str                    # 多头或空头 / Long or short.
    entry: str                   # 信号后下一根开盘 / Open after the signal bar.
    stop: float | None           # 各腿共用止损，否则 None / Shared stop, else None.
    legs: tuple[Leg, ...]        # 部分退出或微仓位，权重和为 1 / Legs with weights summing to 1.
    timeout_bars: int            # 未成交退出腿到期市价退出 / Remaining legs exit at horizon.
    predicted_ev: float          # 净成本加权 EV / Cost-net weighted EV.
    robustness: float            # 政策排序分数 / Policy ranking score.


@dataclass(frozen=True)
class Surface:
    """单根 K 线的校准 EV 矩阵。 / One bar's calibrated EV matrix by side, TP, and SL."""
    ev: np.ndarray
    tp: tuple[float, ...]
    sl: tuple[float, ...]
    horizon_bars: int


# --- 政策 / policies ---------------------------------------------------------

class TradePolicy:
    name = "policy"

    def weights(self, ev: np.ndarray) -> np.ndarray:
        """EV 矩阵映射为非负权重，每根仅一方向、权重和为 1。 / Map EV to one-side, unit-sum weights."""
        raise NotImplementedError

    def score(self, ev: np.ndarray, w: np.ndarray) -> np.ndarray:
        """覆盖率筛选排序分数，默认为政策预测 EV。 / Coverage ranking score; default is predicted EV."""
        return (w * ev).sum((1, 2, 3))

    def build_plan(self, surface: Surface, market_state: dict | None = None,
                   costs: dict | None = None) -> TradePlan:
        """实时候选计划入口；EV 已扣成本，costs 仅为接口稳定性保留，market_state 留给未来状态门槛。

        Live entry point. EV is already net of costs, so `costs` is accepted
        for interface stability; `market_state` is reserved for regime gates."""
        ev = surface.ev[None]
        w = self.weights(ev)
        side = int(w[0].sum((1, 2)).argmax())
        legs = tuple(
            Leg(surface.tp[a], surface.sl[b], float(w[0, side, a, b]))
            for a in range(len(surface.tp)) for b in range(len(surface.sl))
            if w[0, side, a, b] > 1e-12
        )
        stops = {leg.sl for leg in legs}
        return TradePlan(
            side=SIDES[side], entry="next_open",
            stop=stops.pop() if len(stops) == 1 else None, legs=legs,
            timeout_bars=surface.horizon_bars,
            predicted_ev=float((w * ev).sum()),
            robustness=float(self.score(ev, w)[0]),
        )


def _one_hot(shape, flat_idx) -> np.ndarray:
    w = np.zeros((shape[0], int(np.prod(shape[1:]))))
    w[np.arange(shape[0]), flat_idx] = 1.0
    return w.reshape(shape)


def _softmax(x: np.ndarray, tau: float, axis) -> np.ndarray:
    z = x / tau
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class FixedCellPolicy(TradePolicy):
    def __init__(self, grid: dict, tp: float, sl: float):
        self.a = int(np.argmin(np.abs(np.asarray(grid["tp"]) - tp)))
        self.b = int(np.argmin(np.abs(np.asarray(grid["sl"]) - sl)))
        self.name = f"fixed tp{tp:.2%} sl{sl:.2%}"

    def weights(self, ev):
        w = np.zeros_like(ev)
        side = ev[:, :, self.a, self.b].argmax(1)          # 只选方向 / The only choice made.
        w[np.arange(len(ev)), side, self.a, self.b] = 1.0
        return w


class ArgmaxEVPolicy(TradePolicy):
    name = "argmax EV"

    def weights(self, ev):
        return _one_hot(ev.shape, ev.reshape(len(ev), -1).argmax(1))


def local_stats(ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算同方向每单元 3×3 邻域均值和标准差。 / Mean and std of each cell's 3×3 same-side neighbourhood."""
    n, s, a, b = ev.shape
    pad = np.full((n, s, a + 2, b + 2), np.nan)
    pad[:, :, 1:-1, 1:-1] = ev
    stack = np.stack([pad[:, :, i:i + a, j:j + b] for i in range(3) for j in range(3)])
    return np.nanmean(stack, 0), np.nanstd(stack, 0)


class LocalRobustPolicy(TradePolicy):
    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.name = f"local-robust lambda={lam:g}"

    def _cell_score(self, ev):
        mean, std = local_stats(ev)
        return mean - self.lam * std

    def weights(self, ev):
        return _one_hot(ev.shape, self._cell_score(ev).reshape(len(ev), -1).argmax(1))

    def score(self, ev, w):
        return (w * self._cell_score(ev)).sum((1, 2, 3))


class SoftmaxEVPolicy(TradePolicy):
    def __init__(self, tau: float):
        self.tau = tau
        self.name = f"softmax tau={tau * 1e4:g}bps"

    def weights(self, ev):
        n, s, a, b = ev.shape
        w_side = _softmax(ev.reshape(n, s, -1), self.tau, axis=2).reshape(ev.shape)
        side = (w_side * ev).sum((2, 3)).argmax(1)         # 仅二选一，不是四十选一 / Two-way, not 40-way.
        w = np.zeros_like(ev)
        w[np.arange(n), side] = w_side[np.arange(n), side]
        return w


class MultiTPPolicy(TradePolicy):
    """跨 TP 分批退出，共用同一止损。 / Partial TP exits with one shared stop."""

    def __init__(self, tau: float):
        self.tau = tau
        self.name = f"multi-TP tau={tau * 1e4:g}bps"

    def weights(self, ev):
        n, s, a, b = ev.shape
        w_tp = _softmax(ev, self.tau, axis=2)              # 每方向/SL 跨 TP 加权 / Across TPs per side/SL.
        row_ev = (w_tp * ev).sum(2)                        # 每方向/SL 一行 / One row per side/SL.
        best = row_ev.reshape(n, -1).argmax(1)
        side, sl = np.divmod(best, b)
        w = np.zeros_like(ev)
        w[np.arange(n), side, :, sl] = w_tp[np.arange(n), side, :, sl]
        return w


def preregistered(grid: dict) -> list[TradePolicy]:
    """结果出现前固定参数；并列报告，不挑赢家。 / Preregister settings; report all without picking a winner."""
    return [
        FixedCellPolicy(grid, 0.010, 0.010),
        FixedCellPolicy(grid, 0.015, 0.010),
        FixedCellPolicy(grid, 0.020, 0.010),
        ArgmaxEVPolicy(),
        LocalRobustPolicy(0.5), LocalRobustPolicy(1.0), LocalRobustPolicy(2.0),
        SoftmaxEVPolicy(5e-4), SoftmaxEVPolicy(20e-4), SoftmaxEVPolicy(100e-4),
        MultiTPPolicy(5e-4), MultiTPPolicy(20e-4), MultiTPPolicy(100e-4),
    ]


# --- 矩阵与立方体 / matrix and cubes -----------------------------------------

@dataclass
class Cubes:
    ts: np.ndarray
    fold: np.ndarray
    ev: np.ndarray        # 已校准且扣成本的 EV / Calibrated cost-net EV.
    net: np.ndarray       # 每单元已实现净收益 / Realised net return per cell.
    gross: np.ndarray
    dropped: int          # 矩阵不全的 K 线（歧义单元） / Bars lacking full matrices.


def to_cubes(long: pd.DataFrame, grid: dict) -> Cubes:
    tp_codes = np.round(np.asarray(grid["tp"]) * 1e4).astype(int)
    sl_codes = np.round(np.asarray(grid["sl"]) * 1e4).astype(int)
    bars = np.sort(long["ts"].unique())
    i = np.searchsorted(bars, long["ts"].to_numpy())
    s = (long["side"].to_numpy() == "short").astype(int)
    a = np.searchsorted(tp_codes, np.round(long["tp"].to_numpy() * 1e4).astype(int))
    b = np.searchsorted(sl_codes, np.round(long["sl"].to_numpy() * 1e4).astype(int))
    shape = (len(bars), 2, len(tp_codes), len(sl_codes))
    cubes = {}
    for col in ("ev", "net", "gross"):
        c = np.full(shape, np.nan)
        c[i, s, a, b] = long[col].to_numpy()
        cubes[col] = c
    complete = np.isfinite(cubes["ev"]).reshape(len(bars), -1).all(1)
    fold = long.groupby("ts")["fold"].first().reindex(bars).to_numpy()
    return Cubes(bars[complete], fold[complete], cubes["ev"][complete],
                 cubes["net"][complete], cubes["gross"][complete],
                 int((~complete).sum()))


# --- 评估 / evaluation ------------------------------------------------------

def evaluate(policy: TradePolicy, c: Cubes) -> pd.DataFrame:
    w = policy.weights(c.ev)
    return pd.DataFrame({
        "ts": c.ts, "fold": c.fold, "day": c.ts // DAY_MS,
        "pred": (w * c.ev).sum((1, 2, 3)),
        "score": policy.score(c.ev, w),
        "net": (w * c.net).sum((1, 2, 3)),
        "gross": (w * c.gross).sum((1, 2, 3)),
        "side": np.where(w[:, 0].sum((1, 2)) > 0.5, "long", "short"),
        "eff_legs": 1.0 / (w ** 2).sum((1, 2, 3)),
    })


def calibrate_ev(frame: pd.DataFrame, horizon_bars: int = 16,
                 min_train_samples: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """仅用之前的折拟合 net = a + b * pred。 / Fit using strictly earlier folds.

    原始 score 保持不变用于排序；calibrated_pred 仅用于金额解释和报告。

    The original ``score`` remains untouched and is still the selection/ranking
    field.  ``calibrated_pred`` is reporting/economic-value metadata only.
    """
    out = frame.copy()
    out["calibrated_pred"] = np.nan
    rows = []
    ts = out["ts"].to_numpy(np.int64)
    folds = out["fold"].to_numpy()
    for fold in sorted(out["fold"].dropna().unique()):
        test = out[out["fold"] == fold]
        start = int(test["ts"].min())
        prior = (folds < fold) & resolved_before(ts, horizon_bars, start)
        train = out.loc[prior]
        if len(train) < min_train_samples or train["pred"].nunique() < 2 or not len(test):
            continue
        x = train["pred"].to_numpy(float)
        y = train["net"].to_numpy(float)
        a, b = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        if b < 0:
            a, b = float(y.mean()), 0.0
        predicted = a + b * test["pred"].to_numpy(float)
        out.loc[test.index, "calibrated_pred"] = predicted
        error = predicted - test["net"].to_numpy(float)
        rows.append({
            "fold": fold,
            "a": float(a),
            "b": float(b),
            "train_n": len(train),
            "oos_n": len(test),
            "oos_mae": float(np.mean(np.abs(error))),
            "oos_rmse": float(np.sqrt(np.mean(error ** 2))),
            "oos_bias": float(np.mean(error)),
        })
    return out, pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, n_universe: int, universe_days: np.ndarray,
            horizon_bars: int) -> dict[str, Any]:
    """标准化账本：每个选中 K 线开仓权益的 1/horizon，因此总敞口不超过 1 倍。

    Standardised accounting: every selected bar opens 1/horizon of equity, so
    gross exposure never exceeds 1x however densely signals arrive."""
    n = len(frame)
    if n == 0:
        return {"trades": 0}
    r, p = frame["net"].to_numpy(), frame["pred"].to_numpy()
    mean = r.mean()
    cl = pd.Series(r - mean).groupby(frame["day"].to_numpy()).sum().to_numpy()
    se = np.sqrt((cl ** 2).sum()) / n
    f = 1.0 / horizon_bars
    daily = (frame.groupby("day")["net"].sum() * f).reindex(universe_days, fill_value=0.0)
    equity = (1.0 + daily).cumprod()                   # 复利账本 / Compounded book.
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    var_p = p.var()
    if "calibrated_pred" in frame:
        calibrated_mask = frame["calibrated_pred"].notna()
        calibrated = frame.loc[calibrated_mask, "calibrated_pred"]
        calibrated_net = frame.loc[calibrated_mask, "net"]
    else:
        calibrated = pd.Series(dtype=float)
        calibrated_net = pd.Series(dtype=float)
    return {
        "trades": n,
        "coverage": n / n_universe,
        "gross_bps": frame["gross"].mean() * 1e4,
        "fees_bps": (frame["gross"] - frame["net"]).mean() * 1e4,
        "net_bps": mean * 1e4,
        "ci_lo": (mean - 1.96 * se) * 1e4,
        "ci_hi": (mean + 1.96 * se) * 1e4,
        "pred_bps": p.mean() * 1e4,
        "selection_bias_bps": (p.mean() - mean) * 1e4,
        "calibrated_pred_bps": float(calibrated.mean() * 1e4) if len(calibrated) else np.nan,
        "calibrated_bias_bps": float((calibrated - calibrated_net).mean() * 1e4)
        if len(calibrated) else np.nan,
        "calib_slope": float(np.cov(p, r)[0, 1] / var_p) if var_p > 0 else np.nan,
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else np.nan,
        "max_dd_pct": float((equity / equity.cummax() - 1.0).min() * 100),
        "profit_factor": float(wins / losses) if losses > 0 else np.nan,
        "turnover_x_per_day": n * f / len(universe_days),
        "eff_legs": frame["eff_legs"].mean(),
    }


def deciles(frame: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    frame = frame.dropna(subset=[pred_col])
    q = pd.qcut(frame[pred_col], 10, labels=False, duplicates="drop")
    g = frame.groupby(q)
    return pd.DataFrame({"raw_pred_bps": g["pred"].mean() * 1e4,
                         "calibrated_pred_bps": g[pred_col].mean() * 1e4,
                         "realised_bps": g["net"].mean() * 1e4, "n": g.size()})


# --- 报告 / report ---------------------------------------------------------

COVERAGE = (1.0, 0.5, 0.2, 0.1, 0.05)
SHOWCASE = ("fixed tp1.50% sl1.00%", "argmax EV", "local-robust lambda=1",
            "softmax tau=20bps", "multi-TP tau=20bps")


def run_report(data_dir, cfg: dict, model: str = "lgbm",
               out: str = "docs/policy_report.md") -> str:
    from pathlib import Path

    from .audit import _md_table
    from .selective import add_ev, calibrate, matched_oof, realise, select, timeout_prior, universe

    data_dir = Path(data_dir)
    grid, costs, h = cfg["grid"], cfg["costs"], cfg["grid"]["horizon_bars"]
    labels = pd.read_parquet(data_dir / "labels.parquet")
    all_models, match_meta = matched_oof(data_dir, h)
    if model not in all_models:
        raise ValueError(f"model must be one of {sorted(all_models)}")
    long = add_ev(timeout_prior(calibrate(all_models[model], h), labels, h), costs)
    long = realise(long, labels, costs, h)
    full_cubes = to_cubes(long, grid)

    uni = universe(pd.DataFrame({"ts": full_cubes.ts, "fold": full_cubes.fold}), h)
    keep = np.isin(full_cubes.ts, uni["ts"].to_numpy())
    c = Cubes(full_cubes.ts[keep], full_cubes.fold[keep], full_cubes.ev[keep],
              full_cubes.net[keep], full_cubes.gross[keep], full_cubes.dropped)
    days = np.unique(c.ts // DAY_MS)
    n = len(c.ts)

    rows, dec, years, calibration_rows = [], {}, {}, []
    for pol in preregistered(grid):
        full_frame = evaluate(pol, full_cubes)
        full_frame, fitted = calibrate_ev(full_frame, h)
        frame = full_frame[full_frame["ts"].isin(uni["ts"])].copy()
        if len(fitted):
            fitted.insert(0, "policy", pol.name)
            calibration_rows.append(fitted)
        for cov in COVERAGE:
            sel = select(full_frame, "score", cov, "walk_forward", h, eligible=frame)
            rows.append({"policy": pol.name, "target_cov": cov, **metrics(sel, n, days, h)})
        dec[pol.name] = deciles(frame, "calibrated_pred")
        years[pol.name] = frame.groupby(pd.to_datetime(frame["ts"], unit="ms").dt.year)["net"].mean() * 1e4
    table = pd.DataFrame(rows)

    def fmt(df):
        df = df.copy()
        for col in df.columns:
            if df[col].dtype.kind == "f":
                digits = 4 if col == "slope_b" else (3 if col in (
                    "coverage", "calib_slope", "sharpe", "profit_factor",
                    "turnover_x_per_day", "eff_legs", "target_cov") else 1)
                df[col] = df[col].round(digits)
        return df

    all_pred, all_real = np.nanmean(c.ev) * 1e4, np.nanmean(c.net) * 1e4
    cols = ["policy", "trades", "coverage", "gross_bps", "fees_bps", "net_bps", "ci_lo",
            "ci_hi", "pred_bps", "calibrated_pred_bps", "selection_bias_bps",
            "calibrated_bias_bps", "calib_slope", "sharpe",
            "max_dd_pct", "profit_factor", "turnover_x_per_day", "eff_legs"]
    s = [
        f"# Strategy-policy comparison — `{model}` calibrated matrix", "",
        "Hypothesis: treating the TP/SL matrix as a distribution beats hard-selecting "
        "its maximum cell. All policies consume the **same** walk-forward-calibrated "
        "matrix on the **same** bars; every coverage cut-off is walk-forward by each "
        "policy's own score. Parameters were fixed before evaluation and are shown "
        "side by side as a stability check — none is selected.", "",
        "Strategy hierarchy: **fixed cell** is the primary reference; **argmax** is "
        "the comparison baseline; **softmax**, **multi-TP**, and **local robust** "
        "are experimental policies. Threshold, lambda, sizing, regime, and TP/SL "
        "optimization remain frozen. This matched policy sample does not replace "
        "the `INCONCLUSIVE → G3b` ranking gate.", "",
        f"Universe: {n:,} bars with a complete matrix ({c.dropped} dropped for "
        f"ambiguous cells), {len(days)} days. Returns in bps per trade, net of costs. "
        "CIs day-clustered. Sharpe / drawdown / turnover use a standardised book: "
        f"each selected bar opens 1/{h} of equity, so gross exposure stays ≤ 1x.", "",
        f"Matched observation hash: `{match_meta['observation_hash']}`. "
        f"Raw/matched rows: {match_meta['raw_n']} / {match_meta['matched_n']:,}; "
        f"excluded by model: {match_meta['excluded_n']}; "
        f"reasons: {match_meta['excluded_reasons']}.", "",
        "## Reference: the matrix with no selection at all", "",
        f"- Mean predicted EV over every cell of every bar: **{all_pred:.1f} bps**",
        f"- Mean realised net over the same cells: **{all_real:.1f} bps**",
        f"- Gap: **{all_pred - all_real:+.1f} bps** — cell-level EV is unbiased on average, "
        "so any larger gap below is created by the policy's own selection.", "",
    ]
    for cov in COVERAGE:
        t = table[table["target_cov"] == cov][cols]
        s += [f"## Walk-forward coverage {cov:.0%} (ranked by each policy's own score)", "",
              _md_table(fmt(t)), ""]
    s += ["## Calibrated EV vs realised net, by calibrated-EV decile", "",
          "Deciles use only OOS calibrated values. Policy selection and coverage "
          "ranking still use the original `score`; calibration only replaces the EV value.", ""]
    for name, d in dec.items():
        s += [f"### {name}", "", _md_table(fmt(d.reset_index(names="decile"))), ""]
    if calibration_rows:
        calibration = pd.concat(calibration_rows, ignore_index=True)
        calibration = calibration.rename(columns={
            "a": "a_bps", "b": "slope_b",
            "oos_mae": "oos_mae_bps", "oos_rmse": "oos_rmse_bps",
            "oos_bias": "oos_bias_bps",
        })
        for col in ("a_bps", "oos_mae_bps", "oos_rmse_bps", "oos_bias_bps"):
            calibration[col] *= 1e4
        s += ["## Fold-aware linear EV calibration", "",
              "Each policy has its own `a + b × EV_predicted` fit. Fold k uses only "
              "strictly earlier folds; folds without sufficient history are omitted.", "",
              _md_table(fmt(calibration)), ""]
    yr = pd.DataFrame(years).T.round(1)
    yr.columns = [str(col) for col in yr.columns]
    s += ["## Stability — mean realised net by year (bps, 100% coverage)", "",
          _md_table(yr.reset_index(names="policy")), ""]
    text = "\n".join(s) + "\n"
    Path(out).write_text(text, encoding="utf-8")
    return out
