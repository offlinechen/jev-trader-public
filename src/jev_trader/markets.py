"""永续合约公开 K 线适配器，共用严格校验。 / Public perpetual-candle adapters with shared validation."""

from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.parse
import urllib.request

import pandas as pd

from .features import BAR_MS

BINANCE_FAPI = "https://fapi.binance.com"
OKX_API = "https://www.okx.com"
WARMUP_BARS = 400
LIVE_BARS = 1500
_OKX_RATE_LOCK = threading.Lock()
_OKX_LAST_REQUEST = 0.0


def normalize_pair(exchange: str, value: str) -> tuple[str, str]:
    """返回统一交易对和交易所原生合约代码。 / Return CCXT-style pair and exchange-native symbol."""
    exchange = exchange.lower()
    value = value.upper()
    if exchange not in {"binance", "okx"}:
        raise ValueError(f"unsupported exchange {exchange!r}")
    if value.endswith("-SWAP") and exchange == "okx":
        match = re.fullmatch(r"([A-Z0-9]+)-USDT-SWAP", value)
        if not match:
            raise ValueError(f"expected an OKX USDT swap pair, got {value!r}")
        base = match.group(1)
    elif re.fullmatch(r"[A-Z0-9]+/USDT:USDT", value):
        base = value.split("/", 1)[0]
    elif re.fullmatch(r"[A-Z0-9]+USDT", value):
        base = value[:-4]
    else:
        raise ValueError(f"expected a USDT perpetual pair, got {value!r}")
    pair = f"{base}/USDT:USDT"
    native = f"{base}USDT" if exchange == "binance" else f"{base}-USDT-SWAP"
    return pair, native


def _public_json(base_url: str, path: str, params: dict[str, str | int]) -> object:
    url = base_url + path + ("?" + urllib.parse.urlencode(params) if params else "")
    request = urllib.request.Request(url, headers={"User-Agent": "jev-trader/0.1"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _binance_json(path: str, params: dict[str, str | int]) -> object:
    return _public_json(BINANCE_FAPI, path, params)


def _okx_json(path: str, params: dict[str, str | int]) -> object:
    global _OKX_LAST_REQUEST
    with _OKX_RATE_LOCK:
        now = time.monotonic()
        time.sleep(max(0.0, 0.105 - (now - _OKX_LAST_REQUEST)))
        _OKX_LAST_REQUEST = time.monotonic()
    return _public_json(OKX_API, path, params)


def server_time_ms(exchange: str, fetch_json=None) -> int:
    if exchange == "binance":
        payload = (fetch_json or _binance_json)("/fapi/v1/time", {})
        value = payload.get("serverTime") if isinstance(payload, dict) else None
    elif exchange == "okx":
        payload = (fetch_json or _okx_json)("/api/v5/public/time", {})
        data = payload.get("data") if isinstance(payload, dict) else None
        value = data[0].get("ts") if isinstance(data, list) and data else None
    else:
        raise ValueError(f"unsupported exchange {exchange!r}")
    if value is None:
        raise ValueError(f"{exchange} server-time response is malformed")
    return int(value)


def validate_closed_bars(frame: pd.DataFrame, server_ms: int, *,
                         required_bars: int = WARMUP_BARS,
                         exchange: str = "market") -> pd.DataFrame:
    """严格校验 OHLCV，保留可选来源字段。 / Strict OHLCV checks while preserving optional source fields."""
    required = ["ts", "open", "high", "low", "close", "volume"]
    if any(column not in frame for column in required):
        raise ValueError(f"{exchange} candle fields are incomplete")
    frame = frame.copy().sort_values("ts", kind="stable").reset_index(drop=True)
    if len(frame) < required_bars:
        raise ValueError(f"only {len(frame)} closed 15m candles; need {required_bars}")
    values = frame[["open", "high", "low", "close", "volume"]]
    if not values.notna().all().all() or not values.map(math.isfinite).all().all():
        raise ValueError(f"{exchange} 15m candles contain non-finite values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{exchange} 15m candles contain non-positive prices")
    if ((frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
            | (frame["volume"] < 0)).any():
        raise ValueError(f"{exchange} 15m candles contain inconsistent OHLCV")
    if frame["ts"].duplicated().any() or frame["ts"].diff().iloc[1:].ne(BAR_MS).any():
        raise ValueError(f"{exchange} 15m candles are duplicated or have a gap")
    expected = server_ms // BAR_MS * BAR_MS - BAR_MS
    if int(frame.iloc[-1]["ts"]) != expected:
        raise ValueError(f"{exchange} has not published the latest closed 15m candle")
    return frame


def _binance_bars(native_symbol: str, server_ms: int, fetch_json) -> pd.DataFrame:
    raw = fetch_json("/fapi/v1/klines", {
        "symbol": native_symbol, "interval": "15m", "limit": LIVE_BARS,
    })
    if not isinstance(raw, list):
        raise ValueError("Binance klines response is not an array")
    rows = []
    for bar in raw:
        if not isinstance(bar, list) or len(bar) < 10:
            raise ValueError("Binance returned a malformed kline")
        if int(bar[6]) >= server_ms:
            continue
        rows.append({
            "ts": int(bar[0]), "open": float(bar[1]), "high": float(bar[2]),
            "low": float(bar[3]), "close": float(bar[4]),
            "volume": float(bar[5]), "taker_buy_volume": float(bar[9]),
        })
    return validate_closed_bars(pd.DataFrame(rows), server_ms,
                                required_bars=WARMUP_BARS, exchange="Binance")


def _okx_bars(native_symbol: str, server_ms: int, fetch_json,
              max_bars: int = LIVE_BARS) -> pd.DataFrame:
    rows: list[dict] = []
    after: str | None = None
    seen_cursors: set[str] = set()
    while len(rows) < max_bars:
        params: dict[str, str | int] = {"instId": native_symbol, "bar": "15m", "limit": "300"}
        if after is not None:
            params["after"] = after
        payload = fetch_json("/api/v5/market/history-candles", params)
        if not isinstance(payload, dict) or payload.get("code") != "0" or not isinstance(payload.get("data"), list):
            raise ValueError("OKX history-candles response is malformed")
        page = payload["data"]
        if not page:
            break
        for candle in page:
            if not isinstance(candle, list) or len(candle) < 9:
                raise ValueError("OKX returned a malformed candle")
        expected = server_ms // BAR_MS * BAR_MS - BAR_MS
        latest = max(page, key=lambda candle: int(candle[0]))
        if int(latest[0]) == expected and latest[8] != "1":
            raise ValueError("OKX latest closed 15m candle is not confirmed")
        page_rows = []
        for candle in page:
            if candle[8] != "1":
                continue
            ts = int(candle[0])
            if ts + BAR_MS > server_ms:
                continue
            page_rows.append({
                "ts": ts, "open": float(candle[1]), "high": float(candle[2]),
                "low": float(candle[3]), "close": float(candle[4]),
                # 衍生品的 vol 为合约张数，volCcy 为基础币数量。 / Derivatives: vol is contracts; volCcy is base currency.
                "volume": float(candle[6]),
            })
        rows.extend(page_rows)
        # OKX 的 `after` 请求此时间戳之前的数据。 / OKX `after` requests older data.
        next_cursor = str(min(int(row[0]) for row in page))
        if next_cursor in seen_cursors or next_cursor == after:
            raise ValueError("OKX candle pagination did not advance")
        seen_cursors.add(next_cursor)
        after = next_cursor
        if len(page) < 300:
            break
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("OKX returned no closed candles")
    if frame["ts"].duplicated().any():
        raise ValueError("OKX candle pagination returned duplicate timestamps")
    frame = frame.sort_values("ts").tail(max_bars).reset_index(drop=True)
    return validate_closed_bars(frame, server_ms, required_bars=WARMUP_BARS, exchange="OKX")


def fetch_closed_bars(exchange: str, pair: str, server_ms: int, fetch_json=None) -> pd.DataFrame:
    """从公开端点获取并校验连续已收盘 15m K 线。 / Fetch and validate continuous closed 15m candles."""
    exchange = exchange.lower()
    _, native = normalize_pair(exchange, pair)
    if exchange == "binance":
        return _binance_bars(native, server_ms, fetch_json or _binance_json)
    if exchange == "okx":
        return _okx_bars(native, server_ms, fetch_json or _okx_json)
    raise ValueError(f"unsupported exchange {exchange!r}")
