from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[2] / "config" / "jev.yaml"


def load(path: str | Path = DEFAULT) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def to_binance(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTCUSDT' (archive and REST naming)."""
    return symbol.split(":")[0].replace("/", "")
