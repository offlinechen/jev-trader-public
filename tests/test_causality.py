"""T1.2 -- the highest-value test in the repository (NFR-1).

Feature x[t] must be identical whether computed on the full history or on
history[:t+1]. One assertion kills the whole class of lookahead bugs: rolling
windows, multi-timeframe merges, centred indicators, bfill.

It needs no downloaded data -- a synthetic random walk is enough, which keeps
it off the critical path and makes it run in seconds on every commit.

The two `test_detects_*` cases are what make the test trustworthy: they plant
a known leak and demand that the check catches it. A causality check that has
never failed is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from jev_trader.features import BAR_MS, build_features

WARMUP = 400  # longest window: 20 * 16 bars for the 4h EMA


def synthetic(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.0015, n)) * close
    high = np.maximum.reduce([close + wick * rng.random(n), open_, close])
    low = np.minimum.reduce([close - wick * rng.random(n), open_, close])
    vol = rng.lognormal(3, 0.5, n)
    anchor = (1_600_000_000_000 // BAR_MS) * BAR_MS
    return pd.DataFrame(
        {
            "ts": anchor + np.arange(n) * BAR_MS,
            "open": open_, "high": high, "low": low, "close": close,
            "volume": vol, "taker_buy_volume": vol * rng.random(n),
        }
    )


def assert_causal(fn, df: pd.DataFrame, k: int = 40, seed: int = 1) -> None:
    """Rebuild on truncated history at k random bars; demand an exact match."""
    full = fn(df).set_index("ts")
    rng = np.random.default_rng(seed)
    for i in rng.choice(np.arange(WARMUP, len(df)), k, replace=False):
        t = int(df["ts"].iloc[i])
        truncated = fn(df[df["ts"] <= t]).set_index("ts")
        assert_frame_equal(
            full.loc[[t]], truncated.iloc[[-1]], rtol=1e-9, atol=1e-12,
            obj=f"features at ts={t}",
        )


# --- the real thing --------------------------------------------------------

def test_build_features_is_causal():
    assert_causal(build_features, synthetic())


def test_build_features_is_causal_other_seed():
    assert_causal(build_features, synthetic(seed=7), seed=7)


# --- the check has to be able to fail --------------------------------------

def _leaky_shift(df):
    f = build_features(df)
    f["f_peek"] = df["close"].shift(-1).to_numpy() / df["close"].to_numpy() - 1
    return f


def _leaky_centered(df):
    f = build_features(df)
    centered = df["close"].rolling(21, center=True).mean().to_numpy()
    f["f_center"] = centered / df["close"].to_numpy() - 1
    return f


def _leaky_bfill(df):
    f = build_features(df)
    gapped = df["close"].where(df.index % 5 != 0)
    f["f_bfill"] = gapped.bfill().to_numpy() / df["close"].to_numpy() - 1
    return f


@pytest.mark.parametrize("leaky", [_leaky_shift, _leaky_centered, _leaky_bfill])
def test_detects_leak(leaky):
    with pytest.raises(AssertionError):
        assert_causal(leaky, synthetic())


# --- sanity ----------------------------------------------------------------

def test_features_are_populated_after_warmup():
    f = build_features(synthetic()).iloc[WARMUP:]
    empty = [c for c in f.columns if f[c].isna().all()]
    assert not empty, f"all-NaN after warmup: {empty}"


def test_mtf_context_uses_only_closed_bars():
    """The 4h column may only change on the bar that closes a 4h bucket.

    A bucket opening at T closes at T + 16*BAR, which is the close time of the
    15m bar at T + 15*BAR -- so the step lands where (ts + BAR) % 16*BAR == 0.
    """
    f = build_features(synthetic())
    changed = f["f_4h_trend"].diff().ne(0) & f["f_4h_trend"].shift().notna()
    boundary = ((f["ts"] + BAR_MS) % (16 * BAR_MS)) == 0
    offenders = f.loc[changed & ~boundary, "ts"]
    assert offenders.empty, f"4h context moved off-boundary at {list(offenders[:5])}"
