"""Fixed-rule diagnostic signals and a local-only paper execution ledger."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time

from .features import BAR_MS
from .grid import sl_key, tp_key

POLICY_ID = "fixed_tp1_sl1_raw_direction_v1"
FIXED_TP = 0.01
FIXED_SL = 0.01
PAPER_MAX_NOTIONAL_USD = 100.0
PAPER_MAX_RISK_USD = 1.0
ENTRY_GRACE_MS = 60_000


def diagnostic_signal(answers: dict, grid: dict) -> dict:
    """A fixed raw-probability diagnostic; not calibrated EV or a trade claim."""
    if FIXED_TP not in grid["tp"] or FIXED_SL not in grid["sl"]:
        return {"status": "no_signal", "reason": "fixed_cell_missing"}
    scores = {}
    for side in ("long", "short"):
        try:
            p_tp = answers[tp_key(side, FIXED_TP, FIXED_SL)]
            p_sl = answers[sl_key(side, FIXED_TP, FIXED_SL)]
        except KeyError:
            return {"status": "no_signal", "reason": "fixed_cell_missing"}
        if any(isinstance(p, bool) or not isinstance(p, (int, float))
               or not math.isfinite(p) or not 0 <= p <= 1 for p in (p_tp, p_sl)):
            return {"status": "no_signal", "reason": "invalid_fixed_cell_probability"}
        if p_tp + p_sl > 1 + 1e-9:
            return {"status": "no_signal", "reason": "invalid_fixed_cell_probability"}
        scores[side] = float(p_tp - p_sl)

    if math.isclose(scores["long"], scores["short"], abs_tol=1e-12):
        return {"status": "no_signal", "reason": "direction_tie", **_scores(scores)}
    side = max(scores, key=scores.get)
    if scores[side] <= 0:
        return {"status": "no_signal", "reason": "no_positive_fixed_cell_score", **_scores(scores)}
    return {
        "status": "candidate", "reason": "fixed_cell_raw_tp_vs_sl",
        "side": side, "tp": FIXED_TP, "sl": FIXED_SL,
        "policy_id": POLICY_ID, **_scores(scores),
    }


def _scores(scores: dict[str, float]) -> dict:
    return {"score_long": scores["long"], "score_short": scores["short"]}


class PaperEngine:
    """Idempotent local paper execution; never talks to an exchange."""

    def __init__(self, db: sqlite3.Connection, run_id: str, grid: dict, costs: dict):
        self.db, self.run_id, self.grid, self.costs = db, run_id, grid, costs
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS diagnostic_signals (
                run_id TEXT NOT NULL, symbol TEXT NOT NULL, ts INTEGER NOT NULL,
                status TEXT NOT NULL, side TEXT, tp REAL, sl REAL,
                score_long REAL, score_short REAL, policy_id TEXT,
                state_hash TEXT NOT NULL, reason TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                entry_allowed INTEGER NOT NULL DEFAULT 0,
                acceptance_reason TEXT NOT NULL DEFAULT 'legacy_unverified',
                PRIMARY KEY (run_id, symbol, ts)
            );
            CREATE TABLE IF NOT EXISTS dry_orders (
                order_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, symbol TEXT NOT NULL,
                signal_ts INTEGER NOT NULL, entry_ts INTEGER NOT NULL,
                side TEXT NOT NULL, tp REAL NOT NULL, sl REAL NOT NULL,
                notional_cap REAL NOT NULL, risk_cap REAL NOT NULL,
                status TEXT NOT NULL, reason TEXT NOT NULL, created_at_ms INTEGER NOT NULL,
                UNIQUE (run_id, symbol, signal_ts)
            );
            CREATE TABLE IF NOT EXISTS dry_fills (
                order_id TEXT PRIMARY KEY, fill_ts INTEGER NOT NULL,
                fill_price REAL NOT NULL, quantity REAL NOT NULL,
                notional_usd REAL NOT NULL, reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dry_positions (
                order_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, symbol TEXT NOT NULL,
                side TEXT NOT NULL, signal_ts INTEGER NOT NULL, entry_ts INTEGER NOT NULL,
                entry_price REAL NOT NULL, quantity REAL NOT NULL,
                notional_usd REAL NOT NULL, tp_price REAL NOT NULL, sl_price REAL NOT NULL,
                bars_held INTEGER NOT NULL, last_bar_ts INTEGER NOT NULL,
                status TEXT NOT NULL, reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dry_exits (
                order_id TEXT PRIMARY KEY, exit_ts INTEGER NOT NULL,
                exit_price REAL NOT NULL, gross_pnl REAL NOT NULL,
                cost_usd REAL NOT NULL, net_pnl REAL NOT NULL, reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dry_feedback (
                run_id TEXT NOT NULL, symbol TEXT NOT NULL, signal_ts INTEGER NOT NULL,
                order_id TEXT NOT NULL, status TEXT NOT NULL,
                net_pnl REAL, reason TEXT NOT NULL,
                PRIMARY KEY (run_id, symbol, signal_ts)
            );
            CREATE TABLE IF NOT EXISTS dry_state (
                symbol TEXT PRIMARY KEY, last_ts INTEGER NOT NULL
            );
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(diagnostic_signals)")}
        if "entry_allowed" not in columns:
            self.db.execute(
                "ALTER TABLE diagnostic_signals ADD COLUMN entry_allowed "
                "INTEGER NOT NULL DEFAULT 0"
            )
        if "acceptance_reason" not in columns:
            self.db.execute(
                "ALTER TABLE diagnostic_signals ADD COLUMN acceptance_reason "
                "TEXT NOT NULL DEFAULT 'legacy_unverified'"
            )
        self.db.commit()

    def record_no_signal(self, symbol: str, ts: int, state_hash: str, reason: str) -> dict:
        signal = {"status": "no_signal", "reason": reason}
        with self.db:
            self._insert_signal(symbol, ts, state_hash, signal, False, reason)
            self.block_entries(symbol, reason, ts)
            self._advance_watermark(symbol, ts)
        return signal

    def fail_market_gap(self, symbol: str, ts: int) -> dict:
        """Cancel unseen entries and suspend open exposure when candle continuity breaks."""
        self._suspend_for_gap(symbol, ts)
        return self.record_no_signal(symbol, ts, "", "market_data_gap")

    def record_signal(self, symbol: str, ts: int, state_hash: str,
                      answers: dict, *, dry_run: bool = False,
                      allow_entry: bool = True,
                      acceptance_reason: str = "accepted") -> dict:
        signal = diagnostic_signal(answers, self.grid)
        allowed = allow_entry and signal["status"] == "candidate"
        with self.db:
            self._insert_signal(
                symbol, ts, state_hash, signal, allowed,
                acceptance_reason,
            )
            if not allowed:
                self.block_entries(symbol, acceptance_reason, ts)
            stored = self.db.execute(
                "SELECT status, side, tp, sl, reason, policy_id, score_long, score_short, "
                "entry_allowed, acceptance_reason "
                "FROM diagnostic_signals WHERE run_id=? AND symbol=? AND ts=?",
                (self.run_id, symbol, ts),
            ).fetchone()
            if stored and stored[0] == "candidate" and stored[8] and dry_run:
                self._ensure_order(symbol, ts, stored)
                order = self.db.execute(
                    "SELECT status, reason FROM dry_orders WHERE run_id=? AND symbol=? "
                    "AND signal_ts=?", (self.run_id, symbol, ts),
                ).fetchone()
            else:
                order = None
            self._advance_watermark(symbol, ts)
        if stored:
            result = {
                "status": stored[0], "side": stored[1], "tp": stored[2],
                "sl": stored[3], "reason": stored[4], "policy_id": stored[5],
                "score_long": stored[6], "score_short": stored[7],
                "entry_allowed": bool(stored[8]),
                "acceptance_reason": stored[9],
            }
            if order:
                result.update(order_status=order[0], order_reason=order[1])
            return result
        return signal

    def advance(self, symbol: str, bars, *, allow_entries: bool = True) -> list[dict]:
        """Replay unseen closed 15m bars; gaps suspend exposure rather than guess."""
        rows = bars.to_dict("records") if hasattr(bars, "to_dict") else list(bars)
        rows.sort(key=lambda row: int(row["ts"]))
        if not rows:
            return []
        state = self.db.execute(
            "SELECT last_ts FROM dry_state WHERE symbol=?", (symbol,)
        ).fetchone()
        if state is None:
            with self.db:
                self.db.execute(
                    "INSERT INTO dry_state VALUES (?, ?)", (symbol, int(rows[-1]["ts"]))
                )
            return []

        last_ts = int(state[0])
        pending = [row for row in rows if int(row["ts"]) > last_ts]
        events = []
        if pending and int(pending[0]["ts"]) != last_ts + BAR_MS:
            events.extend(self._suspend_for_gap(symbol, int(pending[0]["ts"])))
            last_ts = int(pending[0]["ts"]) - BAR_MS

        for raw in pending:
            bar = {key: float(raw[key]) if key != "ts" else int(raw[key])
                   for key in ("ts", "open", "high", "low", "close")}
            ts = bar["ts"]
            with self.db:
                pending_orders = self.db.execute(
                    "SELECT order_id, run_id, side, tp, sl, notional_cap, risk_cap "
                    "FROM dry_orders WHERE symbol=? AND status='pending' ORDER BY created_at_ms",
                    (symbol,),
                ).fetchall()
                for order in pending_orders:
                    order_id, order_run, side, tp, sl, notional_cap, risk_cap = order
                    if not allow_entries:
                        self.db.execute(
                            "UPDATE dry_orders SET status='cancelled', reason='inference_blocked' "
                            "WHERE order_id=?", (order_id,),
                        )
                        continue
                    order_row = self.db.execute(
                        "SELECT entry_ts FROM dry_orders WHERE order_id=?", (order_id,)
                    ).fetchone()
                    if ts < int(order_row[0]):
                        continue
                    if ts > int(order_row[0]):
                        self.db.execute(
                            "UPDATE dry_orders SET status='cancelled', reason='missed_entry_bar' "
                            "WHERE order_id=?", (order_id,),
                        )
                        events.append({"type": "cancel", "order_id": order_id,
                                       "reason": "missed_entry_bar"})
                        continue
                    active = self.db.execute(
                        "SELECT 1 FROM dry_positions WHERE status IN ('open','suspended') LIMIT 1"
                    ).fetchone()
                    if active:
                        self.db.execute(
                            "UPDATE dry_orders SET status='rejected', reason='single_position_limit' "
                            "WHERE order_id=?", (order_id,),
                        )
                        continue
                    price = bar["open"]
                    quantity = min(notional_cap / price, risk_cap / (price * sl))
                    if not math.isfinite(quantity) or quantity <= 0:
                        self.db.execute(
                            "UPDATE dry_orders SET status='rejected', reason='invalid_size' "
                            "WHERE order_id=?", (order_id,),
                        )
                        continue
                    notional = quantity * price
                    tp_price = price * (1 + tp if side == "long" else 1 - tp)
                    sl_price = price * (1 - sl if side == "long" else 1 + sl)
                    self.db.execute(
                        "UPDATE dry_orders SET status='filled', reason='next_15m_open_assumed' "
                        "WHERE order_id=?", (order_id,),
                    )
                    self.db.execute("INSERT OR IGNORE INTO dry_fills VALUES (?, ?, ?, ?, ?, ?)", (
                        order_id, ts, price, quantity, notional, "next_15m_open_assumed",
                    ))
                    self.db.execute("INSERT OR IGNORE INTO dry_positions VALUES "
                                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 'open', '')", (
                        order_id, order_run, symbol, side,
                        self.db.execute("SELECT signal_ts FROM dry_orders WHERE order_id=?",
                                        (order_id,)).fetchone()[0],
                        ts, price, quantity, notional, tp_price, sl_price, ts,
                    ))
                    events.append({"type": "fill", "order_id": order_id,
                                   "ts": ts, "price": price})

                position = self.db.execute(
                    "SELECT order_id, run_id, signal_ts, side, entry_price, quantity, "
                    "notional_usd, tp_price, sl_price, bars_held "
                    "FROM dry_positions WHERE symbol=? AND status='open' LIMIT 1",
                    (symbol,),
                ).fetchone()
                if position:
                    events.extend(self._update_position(position, bar))
                self.db.execute(
                    "INSERT INTO dry_state VALUES (?, ?) ON CONFLICT(symbol) "
                    "DO UPDATE SET last_ts=excluded.last_ts", (symbol, ts),
                )
            last_ts = ts
        return events

    def _insert_signal(self, symbol: str, ts: int, state_hash: str, signal: dict,
                       entry_allowed: bool, acceptance_reason: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO diagnostic_signals "
                        "(run_id, symbol, ts, status, side, tp, sl, score_long, "
                        "score_short, policy_id, state_hash, reason, created_at_ms, "
                        "entry_allowed, acceptance_reason) VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            self.run_id, symbol, ts, signal["status"], signal.get("side"),
            signal.get("tp"), signal.get("sl"), signal.get("score_long"),
            signal.get("score_short"), signal.get("policy_id"), state_hash,
            signal.get("reason", ""), int(time.time() * 1000),
            int(entry_allowed), acceptance_reason,
        ))

    def block_entries(self, symbol: str, reason: str, ts: int | None = None) -> None:
        if ts is not None:
            self.db.execute(
                "UPDATE diagnostic_signals SET entry_allowed=0, acceptance_reason=? "
                "WHERE run_id=? AND symbol=? AND ts=?",
                (reason, self.run_id, symbol, ts),
            )
        self.db.execute(
            "UPDATE dry_orders SET status='cancelled', reason=? "
            "WHERE symbol=? AND status='pending'", (reason, symbol),
        )

    def block_run_entries(self, reason: str, now_ms: int) -> None:
        """Revoke currently enterable signals and cancel pending orders for this run."""
        latest_ts = now_ms - BAR_MS
        earliest_ts = latest_ts - ENTRY_GRACE_MS
        with self.db:
            self.db.execute(
                "UPDATE diagnostic_signals SET entry_allowed=0, acceptance_reason=? "
                "WHERE run_id=? AND status='candidate' AND entry_allowed=1 "
                "AND ts BETWEEN ? AND ?",
                (reason, self.run_id, earliest_ts, latest_ts),
            )
            self.db.execute(
                "UPDATE dry_orders SET status='cancelled', reason=? "
                "WHERE run_id=? AND status='pending'", (reason, self.run_id),
            )

    def _advance_watermark(self, symbol: str, ts: int) -> None:
        self.db.execute(
            "INSERT INTO dry_state VALUES (?, ?) ON CONFLICT(symbol) DO UPDATE "
            "SET last_ts=MAX(last_ts, excluded.last_ts)", (symbol, ts),
        )

    def _ensure_order(self, symbol: str, signal_ts: int, signal: tuple) -> None:
        _, side, tp, sl, _, policy_id = signal[:6]
        order_id = hashlib.sha256(
            f"{self.run_id}|{symbol}|{signal_ts}|{policy_id}".encode()
        ).hexdigest()[:32]
        active = self.db.execute(
            "SELECT 1 FROM dry_positions WHERE status IN ('open','suspended') "
            "UNION ALL SELECT 1 FROM dry_orders WHERE status='pending' LIMIT 1"
        ).fetchone()
        order_status, order_reason = (
            ("rejected", "single_position_limit") if active else ("pending", "next_bar_open")
        )
        self.db.execute("INSERT OR IGNORE INTO dry_orders VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            order_id, self.run_id, symbol, signal_ts, signal_ts + BAR_MS,
            side, tp, sl, PAPER_MAX_NOTIONAL_USD, PAPER_MAX_RISK_USD,
            order_status, order_reason, int(time.time() * 1000),
        ))

    def _suspend_for_gap(self, symbol: str, next_ts: int) -> list[dict]:
        with self.db:
            self.db.execute(
                "UPDATE dry_orders SET status='cancelled', reason='market_data_gap' "
                "WHERE symbol=? AND status='pending'", (symbol,),
            )
            self.db.execute(
                "UPDATE dry_positions SET status='suspended', reason='market_data_gap' "
                "WHERE symbol=? AND status='open'", (symbol,),
            )
            self.db.execute(
                "INSERT INTO dry_state VALUES (?, ?) ON CONFLICT(symbol) "
                "DO UPDATE SET last_ts=excluded.last_ts", (symbol, next_ts - BAR_MS),
            )
        return [{"type": "market_gap", "symbol": symbol, "ts": next_ts,
                 "reason": "market_data_gap_no_inferred_fill"}]

    def _update_position(self, position: tuple, bar: dict) -> list[dict]:
        order_id, run_id, signal_ts, side, entry, quantity, notional, tp_price, sl_price, bars_held = position
        bars_held = int(bars_held) + 1
        if side == "long":
            sl_hit, tp_hit = bar["low"] <= sl_price, bar["high"] >= tp_price
        else:
            sl_hit, tp_hit = bar["high"] >= sl_price, bar["low"] <= tp_price
        if sl_hit:
            exit_price = sl_price
            reason = "ambiguous_15m_stop_first" if tp_hit else "stop_loss"
        elif tp_hit:
            exit_price = tp_price
            reason = "take_profit"
        elif bars_held >= int(self.grid["horizon_bars"]):
            exit_price = bar["close"]
            reason = f"timeout_{self.grid['horizon_bars']}_bars"
        else:
            self.db.execute(
                "UPDATE dry_positions SET bars_held=?, last_bar_ts=? WHERE order_id=?",
                (bars_held, bar["ts"], order_id),
            )
            return []

        direction = 1.0 if side == "long" else -1.0
        gross_pnl = direction * (exit_price - entry) * quantity
        held_hours = bars_held * 0.25
        cost_rate = (
            2 * self.costs["taker_fee"] + 2 * self.costs["half_spread"]
            + self.costs["slippage"]
            + self.costs["funding_per_8h"] * held_hours / 8
        )
        cost_usd = notional * cost_rate
        net_pnl = gross_pnl - cost_usd
        self.db.execute("UPDATE dry_positions SET bars_held=?, last_bar_ts=?, status='closed', "
                        "reason=? WHERE order_id=?", (
            bars_held, bar["ts"], reason, order_id,
        ))
        self.db.execute("INSERT OR IGNORE INTO dry_exits VALUES (?, ?, ?, ?, ?, ?, ?)", (
            order_id, bar["ts"], exit_price, gross_pnl, cost_usd, net_pnl, reason,
        ))
        self.db.execute("INSERT OR IGNORE INTO dry_feedback VALUES (?, ?, ?, ?, 'closed', ?, ?)", (
            run_id, self.db.execute("SELECT symbol FROM dry_positions WHERE order_id=?",
                                    (order_id,)).fetchone()[0],
            signal_ts, order_id, net_pnl, f"paper_exit:{reason}",
        ))
        return [
            {"type": "exit", "order_id": order_id, "ts": bar["ts"],
             "price": exit_price, "reason": reason, "net_pnl": net_pnl},
            {"type": "feedback", "order_id": order_id, "status": "closed",
             "net_pnl": net_pnl, "reason": f"paper_exit:{reason}"},
        ]
