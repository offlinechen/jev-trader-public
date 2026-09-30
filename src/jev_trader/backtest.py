"""Causal EV selection and execution simulation for Jev predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config, labels as lab
from .grid import project_barrier_surfaces, sl_key, tp_key


@dataclass(frozen=True)
class CellPrior:
    timeout_return: float


def run_backtest(predictions: pd.DataFrame, labels: pd.DataFrame, cfg: dict[str, Any]):
    """Replay predictions in time order; all signal-side estimates are causal."""
    grid, costs, thresholds, risk = cfg["grid"], cfg["costs"], cfg["thresholds"], cfg["risk"]
    if "response_valid" in predictions:
        predictions = predictions[predictions.response_valid.fillna(False)]
    predictions = predictions.sort_values("ts").reset_index(drop=True)
    labels = labels.sort_values("ts").reset_index(drop=True)
    priors = _build_priors(labels, grid)
    label_by_ts = labels.set_index("ts")
    equity = 1.0
    open_until = -1
    trades, decisions = [], []

    for row in predictions.itertuples(index=False):
        ts = int(row.ts)
        if ts < open_until:
            decisions.append({"ts": ts, "action": "blocked_open_position"})
            continue
        if ts not in label_by_ts.index:
            decisions.append({"ts": ts, "action": "no_label"})
            continue
        signal = choose_signal(row, ts, priors, grid, costs, thresholds)
        if signal is None:
            decisions.append({"ts": ts, "action": "none"})
            continue

        side, tp, sl = signal["side"], signal["tp"], signal["sl"]
        label_row = label_by_ts.loc[ts]
        outcome, hold_minutes = _execution_outcome(label_row, side, tp, sl, grid["horizon_bars"])
        if outcome == lab.Outcome.AMBIGUOUS:
            decisions.append({"ts": ts, "action": "ambiguous_no_trade", **signal})
            continue
        gross_return = _gross_return(label_row, outcome, side, tp, sl)
        cost_rate = _round_trip_cost(costs, hold_minutes / 60)
        net_return = gross_return - cost_rate
        stake_frac = min(float(risk["leverage"]), float(risk["risk_frac"]) / sl)
        pnl = equity * stake_frac * net_return
        equity += pnl
        entry_ts = int(label_row["entry_ts"])
        exit_ts = entry_ts + int(hold_minutes * 60_000)
        open_until = exit_ts
        trade = {
            "signal_ts": ts,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "side": side,
            "tp": tp,
            "sl": sl,
            "outcome": outcome.name,
            "hold_minutes": hold_minutes,
            "entry_price": float(label_row["entry_price"]),
            "gross_return": gross_return,
            "cost_rate": cost_rate,
            "net_return": net_return,
            "stake_frac": stake_frac,
            "pnl": pnl,
            "equity": equity,
            **signal,
        }
        trades.append(trade)
        decisions.append({"ts": ts, "action": "trade", **signal})

    return pd.DataFrame(trades), pd.DataFrame(decisions), _metrics(trades, equity)


def run_file(prediction_path: str | Path, cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    data_dir = Path(cfg["data_dir"])
    predictions = pd.read_parquet(prediction_path)
    labels = pd.read_parquet(data_dir / "labels.parquet")
    trades, decisions, metrics = run_backtest(predictions, labels, cfg)
    trades_path = data_dir / "backtest_trades.parquet"
    decisions_path = data_dir / "backtest_decisions.csv"
    report_path = Path("docs/backtest_report.md")
    trades.to_parquet(trades_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    report_path.write_text(
        render_report(prediction_path, predictions, trades, decisions, metrics, cfg),
        encoding="utf-8",
    )
    return trades_path, decisions_path, report_path


def render_report(prediction_path, predictions, trades, decisions, metrics, cfg):
    def metric(name, fmt=""):
        value = metrics.get(name, 0)
        if isinstance(value, float) and math.isinf(value):
            return "∞"
        return format(value, fmt) if fmt else str(value)

    actions = decisions["action"].value_counts().to_dict() if len(decisions) else {}
    sides = trades["side"].value_counts().to_dict() if len(trades) else {}
    return f"""# Jev simulated backtest

Input predictions: `{prediction_path}`  
Market data: real BTCUSDT 15m/1m Binance archives  
Signal bar: t close; entry: t+1 open  
Horizon: {cfg['grid']['horizon_bars']} 15m bars  
Leverage: {cfg['risk']['leverage']}x; risk per trade: {cfg['risk']['risk_frac']:.2%}

## Result

| Metric | Value |
|---|---:|
| Prediction bars | {len(predictions)} |
| Executed trades | {metric('trades')} |
| Simulated entries / exits | {metric('trades')} / {metric('trades')} |
| Buy / sell entries | {sides.get('long', 0)} / {sides.get('short', 0)} |
| Wins / losses | {metric('wins')} / {metric('losses')} |
| Timeouts | {metric('timeouts')} |
| Win rate | {metric('win_rate', '.2%')} |
| Ending equity | {metric('ending_equity', '.6f')} |
| Total return | {metric('total_return', '.2%')} |
| Profit factor | {metric('profit_factor', '.3f')} |
| Max drawdown | {metric('max_drawdown', '.2%')} |
| Median hold | {metric('median_hold_minutes', '.1f')} minutes |
| Mean modeled cost/trade | {metric('mean_cost_rate', '.3%')} |

## Decision flow

- Trades: {actions.get('trade', 0)}
- No signal / EV below threshold or filters: {actions.get('none', 0)}
- Blocked by max-open-trade rule: {actions.get('blocked_open_position', 0)}
- Ambiguous 1m TP/SL ordering: {actions.get('ambiguous_no_trade', 0)}
- Missing labels: {actions.get('no_label', 0)}

## Execution model

- Long and short entries are selected by the highest causal `EV_net`.
- TP/SL are first-touch exits from 1m data; timeout exits at the 16-bar horizon.
- Costs: double taker fee, double half-spread, single slippage, and funding
  prorated by actual holding time.
- Position size is `min(leverage, risk_frac / SL)` so a stop hit risks at most
  the configured fractional equity risk.
- Historical SL/timeout decomposition and timeout return use only labels before
  each signal timestamp. Future labels are used only to replay the exit.

This is a {len(predictions)}-bar inference replay, not the G5 acceptance test. It is not a
statistically sufficient performance claim and does not place exchange orders.
"""


def choose_signal(row, ts, priors, grid, costs, thresholds):
    """Choose the highest causal EV after the entropy/volatility filters."""
    regimes = np.array([getattr(row, f"regime_{name}") for name in ("up", "down", "range", "transition")])
    regime_sum = regimes.sum()
    entropy = _entropy(regimes / regime_sum) if regime_sum > 0 else math.inf
    vol_extreme = float(getattr(row, "vol_extreme"))
    if entropy > thresholds["max_regime_entropy"] or vol_extreme > thresholds["max_vol_extreme"]:
        return None

    candidates = []
    for side in ("long", "short"):
        raw_tp = np.array([
            [float(getattr(row, tp_key(side, tp, sl))) for sl in grid["sl"]]
            for tp in grid["tp"]
        ])
        raw_sl = np.array([
            [float(getattr(row, sl_key(side, tp, sl))) for sl in grid["sl"]]
            for tp in grid["tp"]
        ])
        matrix_tp, matrix_sl, matrix_timeout, matrix_sum = project_barrier_surfaces(raw_tp, raw_sl)
        for i, tp in enumerate(grid["tp"]):
            for j, sl in enumerate(grid["sl"]):
                if not np.isfinite(matrix_sum[i, j]) or matrix_sum[i, j] > 1 + 1e-9:
                    continue
                p_tp = float(matrix_tp[i, j])
                p_sl = float(matrix_sl[i, j])
                prior = _prior_stats(priors[(side, tp, sl)], ts)
                p_timeout = float(matrix_timeout[i, j])
                ev = (
                    p_tp * tp
                    - p_sl * sl
                    + p_timeout * prior.timeout_return
                    - _round_trip_cost(costs, costs["expected_hold_hours"])
                )
                candidates.append({
                    "side": side, "tp": tp, "sl": sl, "p_tp": p_tp,
                    "p_sl": p_sl, "p_timeout": p_timeout, "ev_net": ev,
                    "regime_entropy": entropy, "vol_extreme": vol_extreme,
                })
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: candidate["ev_net"])
    return best if best["ev_net"] > thresholds["min_ev_net"] else None


def _build_priors(labels, grid):
    ts = labels["ts"].to_numpy()
    priors = {}
    for side in ("long", "short"):
        for tp in grid["tp"]:
            for sl in grid["sl"]:
                outcome = lab.outcomes(labels, side, tp, sl, grid["horizon_bars"])
                valid = outcome != lab.Outcome.AMBIGUOUS
                timeout = (outcome == lab.Outcome.TIMEOUT) & valid
                cumulative_timeout = np.cumsum(timeout.astype(int))
                signed_ret = labels["ret_at_horizon"].to_numpy(float)
                if side == "short":
                    signed_ret = -signed_ret
                cumulative_timeout_return = np.cumsum(np.where(timeout, signed_ret, 0.0))
                priors[(side, tp, sl)] = (ts, cumulative_timeout,
                                          cumulative_timeout_return)
    return priors


def _prior_stats(prior, ts):
    timestamps, cumulative_timeout, cumulative_return = prior
    end = int(np.searchsorted(timestamps, ts, side="left"))
    timeout_count = int(cumulative_timeout[end - 1]) if end else 0
    timeout_sum = float(cumulative_return[end - 1]) if end else 0.0
    return CellPrior(
        timeout_return=timeout_sum / timeout_count if timeout_count else 0.0,
    )


def _execution_outcome(row, side, tp, sl, horizon_bars):
    if side == "long":
        tp_t = int(row[lab.level_col(tp, "up")])
        sl_t = int(row[lab.level_col(sl, "dn")])
    else:
        tp_t = int(row[lab.level_col(tp, "dn")])
        sl_t = int(row[lab.level_col(sl, "up")])
    horizon = horizon_bars * lab.BARS_PER_MIN
    tp_hit, sl_hit = 0 <= tp_t < horizon, 0 <= sl_t < horizon
    if tp_hit and sl_hit and tp_t == sl_t:
        return lab.Outcome.AMBIGUOUS, horizon
    if tp_hit and (not sl_hit or tp_t < sl_t):
        return lab.Outcome.TP_FIRST, tp_t + 1
    if sl_hit:
        return lab.Outcome.SL_FIRST, sl_t + 1
    return lab.Outcome.TIMEOUT, horizon


def _gross_return(row, outcome, side, tp, sl):
    if outcome == lab.Outcome.TP_FIRST:
        return tp
    if outcome == lab.Outcome.SL_FIRST:
        return -sl
    if outcome == lab.Outcome.TIMEOUT:
        return float(row["ret_at_horizon"]) * (1 if side == "long" else -1)
    return 0.0


def _round_trip_cost(costs, hold_hours):
    friction = 2 * costs["taker_fee"] + 2 * costs["half_spread"] + costs["slippage"]
    funding = costs["funding_per_8h"] * hold_hours / 8
    return friction + funding


def _entropy(probabilities):
    p = probabilities[probabilities > 0]
    return float(-(p * np.log(p)).sum())


def _metrics(trades, ending_equity):
    if not trades:
        return {"trades": 0, "ending_equity": ending_equity}
    returns = np.array([trade["net_return"] for trade in trades])
    pnls = np.array([trade["pnl"] for trade in trades])
    equity = np.array([trade["equity"] for trade in trades])
    drawdown = equity / np.maximum.accumulate(np.r_[1.0, equity][:-1]) - 1
    gains, losses = pnls[pnls > 0].sum(), -pnls[pnls < 0].sum()
    return {
        "trades": len(trades),
        "wins": int((pnls > 0).sum()),
        "losses": int((pnls < 0).sum()),
        "timeouts": sum(trade["outcome"] == "TIMEOUT" for trade in trades),
        "ending_equity": ending_equity,
        "total_return": ending_equity - 1,
        "win_rate": float((pnls > 0).mean()),
        "profit_factor": float(gains / losses) if losses else math.inf,
        "max_drawdown": float(drawdown.min()),
        "mean_trade_return": float(returns.mean()),
        "mean_cost_rate": float(np.mean([trade["cost_rate"] for trade in trades])),
        "median_hold_minutes": float(np.median([trade["hold_minutes"] for trade in trades])),
    }
