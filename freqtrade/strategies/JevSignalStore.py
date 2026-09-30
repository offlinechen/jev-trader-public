"""持久化诊断信号的最小 Freqtrade 适配器。 / Minimal Freqtrade adapter for persisted diagnostic signals."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy

POLICY_ID = "fixed_tp1_sl1_raw_direction_v1"
BAR_MS = 15 * 60 * 1000
ENTRY_GRACE_MS = 60 * 1000


class JevSignalStore(IStrategy):
    """仅读取获准的候选行，不调用 Jev 或生成信号。 / Read approved rows; no Jev calls or signal generation."""

    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True
    process_only_new_candles = False
    startup_candle_count = 0
    minimal_roi = {"0": 0.01}
    stoploss = -0.01
    use_exit_signal = True
    ignore_buying_expired_candle_after = 60

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def _exchange_matches(self) -> bool:
        """要求明确的交易所身份并拒绝配置漂移。 / Require exchange identity; reject config/env drift."""
        expected = os.environ.get("JEV_SIGNAL_EXCHANGE", "").lower()
        configured = (self.config.get("exchange") or {}).get("name", "").lower()
        return bool(expected and configured and configured == expected)

    def _signal_run_matches(self, db: sqlite3.Connection, run_id: str, pair: str) -> bool:
        expected = tuple(os.environ.get(name, "") for name in (
            "JEV_SIGNAL_EXCHANGE", "JEV_SIGNAL_MARKET_TYPE",
            "JEV_SIGNAL_FEATURE_VERSION", "JEV_SIGNAL_STATE_VERSION",
            "JEV_SIGNAL_MODEL_ID", "JEV_SIGNAL_PROMPT_VERSION",
        ))
        if not all(expected):
            return False
        row = db.execute(
            "SELECT exchange, market_type, feature_version, state_version, "
            "model_id, prompt_version FROM signal_runs WHERE run_id=? AND pair=?",
            (run_id, pair),
        ).fetchone()
        if row is None or tuple(row) != expected:
            return False

        provenance_table = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='signal_run_provenance'"
        ).fetchone()
        provenance = None
        if provenance_table:
            provenance = db.execute(
                "SELECT provenance, excluded_from_research "
                "FROM signal_run_provenance WHERE run_id=?", (run_id,),
            ).fetchone()

        # v4 必须标记真实行情；旧 v3 可缺来源行。 / v4 needs live-market provenance; legacy v3 may lack the row.
        # 明确排除时一律拒绝。 / Explicit exclusions always fail closed.
        is_v4 = expected[-1] == "jev-ohlcv-v4"
        if is_v4:
            return provenance == ("live_market", 0)
        return provenance is None or provenance == ("live_market", 0)

    def leverage(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_leverage: float, max_leverage: float, entry_tag: str | None,
        side: str, **kwargs,
    ) -> float:
        """所有交易所的诊断路径均限制为 1 倍杠杆。 / Keep diagnostic Freqtrade leverage at 1x."""
        return 1.0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        if not self._exchange_matches():
            return dataframe
        run_id = os.environ.get("JEV_SIGNAL_RUN_ID", "")
        db_path = Path(os.environ.get("JEV_SIGNAL_DB", "data/shadow.sqlite"))
        if not run_id or not db_path.is_file() or dataframe.empty:
            return dataframe

        try:
            uri = db_path.resolve().as_uri() + "?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=0.2) as db:
                identity = db.execute(
                    "SELECT identity_value FROM market_identity WHERE identity_key='exchange'"
                ).fetchone()
                if identity is None or identity[0].lower() != os.environ[
                    "JEV_SIGNAL_EXCHANGE"
                ].lower():
                    return dataframe
                if not self._signal_run_matches(db, run_id, metadata["pair"]):
                    return dataframe
                rows = db.execute(
                    "SELECT ts, side FROM diagnostic_signals "
                    "WHERE run_id=? AND symbol=? AND status='candidate' "
                    "AND entry_allowed=1 AND policy_id=?",
                    (run_id, metadata["pair"], POLICY_ID),
                ).fetchall()
        except sqlite3.Error:
            return dataframe

        signals = {int(ts): side for ts, side in rows}
        candle_ts = (
            pd.to_datetime(dataframe["date"], utc=True)
            .astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
        )
        selected = candle_ts.map(signals)
        tags = candle_ts.map(lambda ts: f"jev_fixed_1pct:{int(ts)}")
        long_mask, short_mask = selected.eq("long"), selected.eq("short")
        dataframe.loc[long_mask, "enter_long"] = 1
        dataframe.loc[long_mask, "enter_tag"] = tags[long_mask]
        dataframe.loc[short_mask, "enter_short"] = 1
        dataframe.loc[short_mask, "enter_tag"] = tags[short_mask]
        return dataframe

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime, entry_tag: str | None,
        side: str, **kwargs,
    ) -> bool:
        """Freqtrade 提交入场前再次核对精确信号。 / Recheck the exact signal immediately before entry."""
        try:
            prefix, raw_ts = (entry_tag or "").rsplit(":", 1)
            signal_ts = int(raw_ts)
            if prefix != "jev_fixed_1pct" or signal_ts % BAR_MS:
                return False
            if current_time.tzinfo is None:
                return False
            now_ms = int(current_time.astimezone(timezone.utc).timestamp() * 1000)
            age_ms = now_ms - signal_ts
            if not BAR_MS <= age_ms <= BAR_MS + ENTRY_GRACE_MS:
                return False
            if not self._exchange_matches():
                return False

            run_id = os.environ.get("JEV_SIGNAL_RUN_ID", "")
            db_path = Path(os.environ.get("JEV_SIGNAL_DB", "data/shadow.sqlite"))
            if not run_id or not db_path.is_file():
                return False
            uri = db_path.resolve().as_uri() + "?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=0.2) as db:
                identity = db.execute(
                    "SELECT identity_value FROM market_identity WHERE identity_key='exchange'"
                ).fetchone()
                if identity is None or identity[0].lower() != os.environ[
                    "JEV_SIGNAL_EXCHANGE"
                ].lower():
                    return False
                if not self._signal_run_matches(db, run_id, pair):
                    return False
                row = db.execute(
                    "SELECT side FROM diagnostic_signals "
                    "WHERE run_id=? AND symbol=? AND ts=? "
                    "AND status='candidate' AND entry_allowed=1 AND policy_id=?",
                    (run_id, pair, signal_ts, POLICY_ID),
                ).fetchone()
            return row is not None and row[0] == side
        except (ValueError, TypeError, OverflowError, OSError, sqlite3.Error):
            return False

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(
        self, pair: str, trade, current_time: datetime, current_rate: float,
        current_profit: float, **kwargs,
    ) -> str | None:
        if current_time >= trade.open_date_utc + timedelta(minutes=15 * 16):
            return "timeout_16_bars"
        return None
