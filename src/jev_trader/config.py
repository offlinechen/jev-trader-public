from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[2] / "config" / "jev.yaml"


def load(path: str | Path = DEFAULT) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def to_binance(symbol: str) -> str:
    """将统一交易对转为 Binance 归档/REST 代码。 / Convert a pair to Binance archive/REST naming."""
    return symbol.split(":")[0].replace("/", "")
