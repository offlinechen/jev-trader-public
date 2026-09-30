from __future__ import annotations

import pytest

from jev_trader.features import BAR_MS
from jev_trader.markets import fetch_closed_bars, normalize_pair, validate_closed_bars


def _okx_rows(server_ms: int, count: int = 500):
    last = server_ms // BAR_MS * BAR_MS - BAR_MS
    return [[str(last - i * BAR_MS), "100", "102", "99", "101", "12",
             "0.12", "12.12", "1"] for i in range(count)]


def test_pair_normalization_is_exchange_explicit():
    assert normalize_pair("binance", "BTCUSDT") == ("BTC/USDT:USDT", "BTCUSDT")
    assert normalize_pair("okx", "BTC/USDT:USDT") == (
        "BTC/USDT:USDT", "BTC-USDT-SWAP"
    )
    with pytest.raises(ValueError, match="exchange"):
        normalize_pair("unknown", "BTCUSDT")


def test_okx_history_paginates_and_uses_base_volume_and_ms_timestamps():
    server_ms = 1_800_000_000_000
    calls = []

    def fetch(path, params):
        calls.append((path, dict(params)))
        rows = _okx_rows(server_ms, 300)
        if "after" in params:
            cursor = int(params["after"])
            rows = [[str(cursor - (i + 1) * BAR_MS), *row[1:]]
                    for i, row in enumerate(rows)]
        return {"code": "0", "data": rows}

    frame = fetch_closed_bars("okx", "BTC/USDT:USDT", server_ms, fetch)
    assert len(calls) == 5
    assert all(path == "/api/v5/market/history-candles" for path, _ in calls)
    assert all(params["instId"] == "BTC-USDT-SWAP" for _, params in calls)
    assert all(params.get("after") for _, params in calls[1:])
    assert frame.iloc[-1]["ts"] == server_ms // BAR_MS * BAR_MS - BAR_MS
    assert frame.iloc[-1]["volume"] == pytest.approx(0.12)
    assert "taker_buy_volume" not in frame


def test_okx_unconfirmed_latest_bar_is_excluded_but_partial_data_fails_closed():
    server_ms = 1_800_000_000_000
    rows = _okx_rows(server_ms, 300)
    rows[0][8] = "0"

    def fetch(_path, _params):
        return {"code": "0", "data": rows}

    with pytest.raises(ValueError, match="latest closed"):
        fetch_closed_bars("okx", "BTC/USDT:USDT", server_ms, fetch)


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: rows.__setitem__(1, rows[0]), "duplicated"),
    (lambda rows: rows.__setitem__(2, [str(int(rows[2][0]) - BAR_MS), *rows[2][1:]]), "duplicated or have a gap"),
    (lambda rows: rows[0].__setitem__(4, "nan"), "non-finite"),
])
def test_common_validation_rejects_bad_bars(mutate, message):
    server_ms = 1_800_000_000_000
    rows = _okx_rows(server_ms, count=401)
    rows.reverse()
    mutate(rows)
    import pandas as pd

    frame = pd.DataFrame([{
        "ts": int(row[0]), "open": float(row[1]), "high": float(row[2]),
        "low": float(row[3]), "close": float(row[4]), "volume": float(row[6]),
    } for row in rows])
    with pytest.raises(ValueError, match=message):
        validate_closed_bars(frame, server_ms, required_bars=400, exchange="okx")


def test_binance_adapter_keeps_taker_buy_volume_compatibility():
    server_ms = 1_800_000_000_000
    raw = []
    last = server_ms // BAR_MS * BAR_MS - BAR_MS
    for i in range(400):
        ts = last - (399 - i) * BAR_MS
        raw.append([ts, "100", "102", "99", "101", "10", ts + BAR_MS - 1,
                    "1000", 1, "4", "400", "0"])

    def fetch(_path, _params):
        return raw

    frame = fetch_closed_bars("binance", "BTCUSDT", server_ms, fetch)
    assert len(frame) == 400
    assert frame.iloc[-1]["taker_buy_volume"] == 4
