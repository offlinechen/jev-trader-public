"""单次真实付费请求契约测试；仅 JEV_LIVE=1 时运行。 / Opt-in one-request live contract test."""

from __future__ import annotations

import os

import pytest

from jev_trader import config
from jev_trader.jev import JevClient, JevSettings, build_questions


@pytest.mark.skipif(os.getenv("JEV_LIVE") != "1", reason="set JEV_LIVE=1 to spend one API request")
def test_live_decisions_contract():
    cfg = config.load()
    settings = JevSettings.from_config(cfg)
    questions = build_questions(cfg["grid"])
    result = JevClient(settings).decide(
        {
            "symbol": cfg["symbols"][0],
            "timeframe": cfg["timeframe"],
            "ts": 1754006400000,
            "features": {"close": 100000.0, "return_1": 0.001, "volatility": 0.02},
        },
        questions,
        use_cache=False,
    )
    assert len(result.answers) == 88
    expected = {f"regime_{name}" for name in ("up", "down", "range", "transition")}
    expected |= {f"vol_{name}" for name in ("low", "normal", "high", "extreme")}
    expected |= {key for key in questions if key.startswith("p_")}
    assert set(result.answers) == expected
    assert all(0 <= value <= 1 for value in result.answers.values())
    assert result.usage["input_tokens"] >= 0
    assert result.usage["output_tokens"] >= 0
