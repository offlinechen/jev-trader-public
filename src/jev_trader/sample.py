"""Stratified bar sampling for the Jev spike (T3.2).

Never a contiguous block. A 2k-bar run drawn from one stretch of tape measures
one market, and the spike's whole job is to detect signal that generalises.

Strata: year x quarter x volatility quartile x direction tercile. Allocation is
proportional to the population, so the sample preserves the real mix rather
than manufacturing a balanced one -- with a floor so thin strata still appear.

Regimes are derived from features, not from Jev: the sample has to be fixed
before any inference runs, or the evaluation is circular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOL_COL = "f_atr_pct"
DIR_COL = "f_4h_trend"


def strata(features: pd.DataFrame) -> pd.DataFrame:
    f = features.dropna(subset=[VOL_COL, DIR_COL]).copy()
    ts = pd.to_datetime(f["ts"], unit="ms")
    f["year"] = ts.dt.year
    f["quarter"] = ts.dt.year.astype(str) + "Q" + ts.dt.quarter.astype(str)
    f["vol_q"] = pd.qcut(f[VOL_COL], 4, labels=["v1", "v2", "v3", "v4"])
    f["dir_t"] = pd.qcut(f[DIR_COL], 3, labels=["down", "flat", "up"])
    f["stratum"] = (
        f["year"].astype(str) + "|" + f["quarter"] + "|"
        + f["vol_q"].astype(str) + "|" + f["dir_t"].astype(str)
    )
    return f


def stratified_sample(
    features: pd.DataFrame, eligible_ts: np.ndarray, n: int = 2000,
    seed: int = 0, floor: int = 5,
) -> pd.DataFrame:
    """Draw ~n bars, proportional across strata, restricted to `eligible_ts`.

    `eligible_ts` is the set of bars with OOF baseline predictions -- the spike
    must be comparable against the baselines on exactly the same observations.
    """
    f = strata(features)
    f = f[f["ts"].isin(eligible_ts)]
    if len(f) < n:
        raise ValueError(f"only {len(f)} eligible bars for a sample of {n}")

    rng = np.random.default_rng(seed)
    share = f["stratum"].value_counts(normalize=True)
    picks = []
    for s, frac in share.items():
        pool = f.index[f["stratum"] == s].to_numpy()
        k = min(len(pool), max(floor, int(round(frac * n))))
        picks.append(rng.choice(pool, k, replace=False))
    out = f.loc[np.concatenate(picks)].sort_values("ts")

    # Trim proportionally if the floor overshot, keeping strata represented.
    if len(out) > n:
        out = out.groupby("stratum", group_keys=False, observed=True).apply(
            lambda g: g.sample(max(floor, int(round(len(g) * n / len(out)))),
                               random_state=seed)
            if len(g) > floor else g
        ).sort_values("ts")
    return out


def composition(sample: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Sample vs population share per stratum dimension, with the skew ratio.

    T3.2's check: no stratum under-represented by more than 2x.
    """
    pop = strata(features)
    rows = []
    for dim in ("year", "quarter", "vol_q", "dir_t"):
        s = sample[dim].value_counts(normalize=True)
        p = pop[dim].value_counts(normalize=True)
        for k in p.index:
            rows.append({
                "dim": dim, "level": str(k),
                "population": float(p.get(k, 0)), "sample": float(s.get(k, 0)),
                "ratio": float(s.get(k, 0) / p[k]) if p[k] else np.nan,
            })
    df = pd.DataFrame(rows)
    df["ok"] = df["ratio"].between(0.5, 2.0)
    return df
