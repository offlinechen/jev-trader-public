"""Selective-trading evaluation: does "trade less, trade better" hold?

Pipeline, all walk-forward and all causal:

1. **Calibrate.** Per fold k, fit one isotonic map per class (TP / SL / TIMEOUT)
   on OOF rows from earlier folds *whose labels had resolved* before fold k
   started, apply to fold k, renormalise the three to sum to 1. Raw model
   probabilities never reach EV.
2. **Timeout economics.** E[terminal return | TIMEOUT] per (side, tp, sl) from the
   same resolved-prior rows. Timeout is not neutral and not a loss.
3. **EV per cell.** `c_tp*TP - c_sl*SL + c_to*E[r|timeout] - cost(expected hold)`.
4. **Best action per bar.** argmax EV over all 40 cells; also the best EV on each
   side, so the margin over the opposite side is known.
5. **Realise.** The chosen cell's actual first-touch outcome: +TP, -SL, or the
   signed terminal return at the horizon, minus cost at the *actual* holding time.

Then coverage curves, breakdowns and model-agreement buckets on that per-bar table.

Two coverage modes, and only one of them is a trading result:

* `diagnostic` ranks bars against the whole evaluation period. It measures
  ranking quality -- whether better scores mean better trades -- but its
  cut-off uses the future distribution of scores, so it is **not tradeable**.
* `walk_forward` sets each fold's cut-off from the score quantile of earlier
  folds only. Realised coverage drifts from the target; that drift is real and
  is reported. This is the number to trade on.

Uncertainty is day-clustered throughout. The 40 cells of one bar are one price
path, and 4h horizons make neighbouring bars share most of their path, so
independent-observation standard errors would be several times too small.
`ess` is the resulting effective sample size.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import labels as lab
from .backtest import _round_trip_cost
from .splits import BAR_MS, resolved_before

MIN_MS = 60_000
DAY_MS = 86_400_000
COVERAGE = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01)
PROB = ("p_sl_first", "p_tp_first", "p_timeout")          # index == Outcome value


# --- inputs -----------------------------------------------------------------

def load_oof(data_dir: str | Path, model: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(data_dir) / "oof", filters=[("model", "==", model)])
    df = df[["ts", "side", "tp", "sl", "fold", "outcome", *PROB]].copy()
    df["side"] = df["side"].astype(str)
    df[["tp", "sl"]] = df[["tp", "sl"]].astype(float).round(6)
    return df


def from_jev_long(path: str | Path, fold_of_ts: pd.Series) -> pd.DataFrame:
    """Adapt g3a's long Jev frame to the OOF schema, attaching baseline folds.

    Rows whose raw TP+SL exceed 1 carry a negative implied timeout; they are
    invalid probability vectors and are dropped, not repaired.
    """
    j = pd.read_parquet(path).rename(columns={"p_tp": "p_tp_first", "p_sl": "p_sl_first"})
    invalid = j["p_timeout"] < 0
    if "prob_invalid" in j:
        invalid |= j["prob_invalid"].astype(bool)
    bad_ts = set(j.loc[invalid, "ts"])
    j = j[~j["ts"].isin(bad_ts)]
    j = j[(j["p_timeout"] >= 0) & (j["outcome"] != int(lab.Outcome.AMBIGUOUS))]
    j["fold"] = j["ts"].map(fold_of_ts)
    j = j.dropna(subset=["fold"]).astype({"fold": int})
    j[["tp", "sl"]] = j[["tp", "sl"]].astype(float).round(6)
    return j[
        ["ts", "side", "tp", "sl", "fold", "outcome", *PROB]
    ]


MATCH_KEYS = ["ts", "side", "tp", "sl", "fold", "outcome"]


def _observation_hash(frame: pd.DataFrame) -> str:
    keys = frame[MATCH_KEYS].sort_values(MATCH_KEYS).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest()


def _stack_oof(lgbm: pd.DataFrame, jev: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Build fold-OOS stack probabilities from the two registered inputs."""
    keys = MATCH_KEYS
    merged = lgbm.merge(jev, on=keys, suffixes=("_base", "_jev"), validate="one_to_one")
    rows = []
    for fold, start in merged.groupby("fold")["ts"].min().sort_index().items():
        ts = merged["ts"].to_numpy()
        prior = (merged["fold"].to_numpy() < fold) & resolved_before(ts, horizon_bars, start)
        train = merged.loc[prior & merged["outcome"].isin([0, 1])]
        test = merged[merged["fold"] == fold]
        if len(train) < 2 or train["outcome"].nunique() < 2 or not len(test):
            continue

        def conditional(frame, suffix):
            tp = frame[f"p_tp_first{suffix}"].to_numpy(float)
            sl = frame[f"p_sl_first{suffix}"].to_numpy(float)
            return tp / np.maximum(tp + sl, 1e-12)

        def logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        x_train = np.column_stack([logit(conditional(train, "_base")),
                                    logit(conditional(train, "_jev"))])
        x_test = np.column_stack([logit(conditional(test, "_base")),
                                   logit(conditional(test, "_jev"))])
        model = LogisticRegression(max_iter=1000).fit(x_train, (train["outcome"] == 1).astype(int))
        p_tp_sl = model.predict_proba(x_test)[:, 1]
        timeout = test["p_timeout_base"].to_numpy(float)
        rows.append(test[keys].assign(
            p_sl_first=(1 - p_tp_sl) * (1 - timeout),
            p_tp_first=p_tp_sl * (1 - timeout),
            p_timeout=timeout,
        ))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=keys + list(PROB))


def matched_oof(data_dir: str | Path, horizon_bars: int) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load lgbm/Jev/stack on one strict, order-independent observation set."""
    data_dir = Path(data_dir)
    lgbm = load_oof(data_dir, "lgbm")
    fold_of_ts = lgbm.drop_duplicates("ts").set_index("ts")["fold"]
    jev_path = data_dir / "g3a_long.parquet"
    if not jev_path.is_file():
        raise FileNotFoundError(f"missing Jev artifact: {jev_path}")
    jev = from_jev_long(jev_path, fold_of_ts)
    stack = _stack_oof(lgbm, jev, horizon_bars)
    raw = {"lgbm": lgbm, "jev": jev, "stack": stack}
    sets = {name: set(map(tuple, frame[MATCH_KEYS].itertuples(index=False, name=None)))
            for name, frame in raw.items()}
    common = set.intersection(*sets.values())
    matched = {}
    for name, frame in raw.items():
        keep = frame[MATCH_KEYS].apply(tuple, axis=1).isin(common)
        matched[name] = frame.loc[keep].sort_values(MATCH_KEYS).reset_index(drop=True)
    key_frame = matched["lgbm"][MATCH_KEYS]
    meta = {
        "raw_n": {name: len(frame) for name, frame in raw.items()},
        "matched_n": len(key_frame),
        "excluded_n": {name: len(frame) - len(key_frame) for name, frame in raw.items()},
        "excluded_reasons": {
            "lgbm": {
                "not_in_jev": len(sets["lgbm"] - sets["jev"]),
                "not_in_stack": len(sets["lgbm"] - sets["stack"]),
            },
            "jev": {
                "not_in_lgbm": len(sets["jev"] - sets["lgbm"]),
                "not_in_stack": len(sets["jev"] - sets["stack"]),
            },
            "stack": {
                "not_in_lgbm": len(sets["stack"] - sets["lgbm"]),
                "not_in_jev": len(sets["stack"] - sets["jev"]),
            },
        },
        "observation_hash": _observation_hash(key_frame),
    }
    return matched, meta


# --- steps 1-3 --------------------------------------------------------------

def calibrate(df: pd.DataFrame, horizon_bars: int, min_prior_folds: int = 3,
              max_fit: int = 500_000, seed: int = 0) -> pd.DataFrame:
    ts, fold, y = df["ts"].to_numpy(), df["fold"].to_numpy(), df["outcome"].to_numpy()
    raw = df[list(PROB)].to_numpy(float)
    cal = np.full(raw.shape, np.nan)
    rng = np.random.default_rng(seed)
    for k, start in df.groupby("fold")["ts"].min().sort_index().items():
        prior = (fold < k) & resolved_before(ts, horizon_bars, start)
        if len(np.unique(fold[prior])) < min_prior_folds:
            continue
        idx = np.flatnonzero(prior)
        # ponytail: subsample the fit set; isotonic on 500k rows is already
        # far past the resolution a 10-bin reliability check can see.
        if len(idx) > max_fit:
            idx = rng.choice(idx, max_fit, replace=False)
        cur = fold == k
        for c in range(3):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw[idx, c], (y[idx] == c).astype(float))
            cal[cur, c] = iso.predict(raw[cur, c])
    total = cal.sum(1, keepdims=True)
    cal = cal / np.where(total > 0, total, np.nan)
    out = df.assign(c_sl=cal[:, 0], c_tp=cal[:, 1], c_to=cal[:, 2])
    return out[out["c_tp"].notna()].copy()


def timeout_prior(df: pd.DataFrame, labels: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    df = df.merge(labels[["ts", "ret_at_horizon"]], on="ts", how="left")
    signed = np.where(df["side"] == "long", 1.0, -1.0) * df["ret_at_horizon"].to_numpy()
    ts, fold = df["ts"].to_numpy(), df["fold"].to_numpy()
    is_to = df["outcome"].to_numpy() == int(lab.Outcome.TIMEOUT)
    e_to = np.full(len(df), np.nan)
    keys = ["side", "tp", "sl"]
    for k, start in df.groupby("fold")["ts"].min().sort_index().items():
        m = (fold < k) & is_to & resolved_before(ts, horizon_bars, start)
        if not m.any():
            continue
        means = (df.loc[m, keys].assign(r=signed[m]).groupby(keys)["r"].mean()
                 .rename("e").reset_index())
        cur = np.flatnonzero(fold == k)
        e_to[cur] = df.iloc[cur][keys].merge(means, on=keys, how="left")["e"].to_numpy()
    return df.assign(e_to=np.nan_to_num(e_to, nan=0.0))


def add_ev(df: pd.DataFrame, costs: dict) -> pd.DataFrame:
    cost = _round_trip_cost(costs, costs["expected_hold_hours"])
    ev = df["c_tp"] * df["tp"] - df["c_sl"] * df["sl"] + df["c_to"] * df["e_to"] - cost
    conf = df["c_tp"] / (df["c_tp"] + df["c_sl"]).replace(0, np.nan)
    return df.assign(ev=ev, conf=conf)


# --- steps 4-5 --------------------------------------------------------------

def realise(frame: pd.DataFrame, labels: pd.DataFrame, costs: dict, horizon_bars: int) -> pd.DataFrame:
    """Actual first-touch payoff of every row's (side, tp, sl) cell.

    +TP, -SL, or the signed terminal return at the horizon; cost charged at the
    *actual* holding time. Works on the full long matrix or on any subset of it,
    so every policy is realised by exactly the same code.
    """
    touch = ["ts"] + [c for c in labels.columns if c.startswith(("up_", "dn_"))]
    out = frame.reset_index(drop=True).merge(labels[touch], on="ts", how="left")
    horizon_min = horizon_bars * lab.BARS_PER_MIN
    hold = np.full(len(out), horizon_min, dtype=float)
    gross = np.zeros(len(out))
    y = out["outcome"].to_numpy()
    for (side, tp, sl), g in out.groupby(["side", "tp", "sl"]).groups.items():
        g = np.asarray(g)
        tp_t = out.loc[g, lab.level_col(tp, "up" if side == "long" else "dn")].to_numpy()
        sl_t = out.loc[g, lab.level_col(sl, "dn" if side == "long" else "up")].to_numpy()
        sign = 1.0 if side == "long" else -1.0
        yy = y[g]
        gross[g] = np.select(
            [yy == lab.Outcome.TP_FIRST, yy == lab.Outcome.SL_FIRST],
            [tp, -sl], sign * out.loc[g, "ret_at_horizon"].to_numpy())
        hold[g] = np.select(
            [yy == lab.Outcome.TP_FIRST, yy == lab.Outcome.SL_FIRST],
            [tp_t + 1, sl_t + 1], horizon_min)
    friction = 2 * costs["taker_fee"] + 2 * costs["half_spread"] + costs["slippage"]
    cost = friction + costs["funding_per_8h"] * hold / 60 / 8   # == _round_trip_cost, vectorised
    return out.drop(columns=touch[1:]).assign(hold_min=hold, gross=gross, net=gross - cost)


def per_bar(df: pd.DataFrame, labels: pd.DataFrame, costs: dict, horizon_bars: int) -> pd.DataFrame:
    best = df.loc[df.groupby("ts")["ev"].idxmax()].set_index("ts")
    by_side = df.groupby(["ts", "side"])["ev"].max().unstack()
    best["ev_long"] = by_side.get("long")
    best["ev_short"] = by_side.get("short")
    other = np.where(best["side"] == "long", best["ev_short"], best["ev_long"])
    best["margin"] = best["ev"] - other
    best = realise(best.reset_index(), labels, costs, horizon_bars)
    best["win"] = best["outcome"].to_numpy() == lab.Outcome.TP_FIRST
    best["day"] = best["ts"] // DAY_MS
    best["entry_ts"] = best["ts"] + BAR_MS
    best["exit_ts"] = best["entry_ts"] + best["hold_min"] * MIN_MS
    keep = ["ts", "fold", "day", "side", "tp", "sl", "outcome", "win", "c_tp", "c_sl",
            "c_to", "e_to", "ev", "ev_long", "ev_short", "margin", "conf",
            "hold_min", "gross", "net", "entry_ts", "exit_ts"]
    return best[keep]


# --- statistics -------------------------------------------------------------

def summarise(bars: pd.DataFrame) -> dict:
    """Mean realised net return with a day-clustered CI and effective sample size."""
    n = len(bars)
    if n == 0:
        return {"trades": 0, "days": 0, "ess": 0.0, "win_rate": np.nan, "timeout_rate": np.nan,
                "mean_pred_ev": np.nan, "mean_net": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    r = bars["net"].to_numpy()
    mean = r.mean()
    cluster = pd.Series(r - mean).groupby(bars["day"].to_numpy()).sum().to_numpy()
    var_mean = (cluster ** 2).sum() / n ** 2
    s2 = r.var(ddof=1) if n > 1 else np.nan
    se = np.sqrt(var_mean)
    return {
        "trades": n,
        "days": int(bars["day"].nunique()),
        "ess": float(s2 / var_mean) if var_mean > 0 else float(n),
        "win_rate": float(bars["win"].mean()),
        "timeout_rate": float((bars["outcome"] == lab.Outcome.TIMEOUT).mean()),
        "mean_pred_ev": float(bars["ev"].mean()),
        "mean_net": float(mean),
        "ci_lo": float(mean - 1.96 * se),
        "ci_hi": float(mean + 1.96 * se),
    }


def non_overlapping(bars: pd.DataFrame) -> pd.DataFrame:
    """One position at a time: skip a signal while the previous trade is open."""
    s = bars.sort_values("ts")
    keep, busy_until = [], -1
    for i, entry, exit_ in zip(s.index, s["entry_ts"].to_numpy(), s["exit_ts"].to_numpy()):
        if entry >= busy_until:
            keep.append(i)
            busy_until = exit_
    return s.loc[keep]


def universe(bars: pd.DataFrame, horizon_bars: int, min_prior_folds: int = 3) -> pd.DataFrame:
    """Bars in folds that have enough resolved history to set a causal cut-off.

    Both coverage modes are evaluated on exactly this set, so the only
    difference between them is where the cut-off comes from.
    """
    ok = []
    for k, start in bars.groupby("fold")["ts"].min().sort_index().items():
        prior = bars["fold"].lt(k) & resolved_before(bars["ts"].to_numpy(), horizon_bars, start)
        if bars.loc[prior, "fold"].nunique() >= min_prior_folds:
            ok.append(k)
    return bars[bars["fold"].isin(ok)]


def select(bars: pd.DataFrame, score: str, coverage: float, mode: str,
           horizon_bars: int, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Top `coverage` of the universe by `score`.

    diagnostic: one cut-off from the whole universe (uses the future; ranking only).
    walk_forward: each fold's cut-off from resolved earlier bars only (tradeable).
    """
    uni = universe(bars, horizon_bars) if eligible is None else bars[bars["ts"].isin(eligible["ts"])]
    if coverage >= 1.0:
        return uni
    if mode == "diagnostic":
        return uni[uni[score] >= uni[score].quantile(1 - coverage)]
    picked = []
    for k, start in uni.groupby("fold")["ts"].min().sort_index().items():
        prior = bars[bars["fold"].lt(k) & resolved_before(bars["ts"].to_numpy(), horizon_bars, start)]
        cur = uni[uni["fold"] == k]
        picked.append(cur[cur[score] >= prior[score].quantile(1 - coverage)])
    return pd.concat(picked)


def coverage_curve(bars: pd.DataFrame, score: str, mode: str, horizon_bars: int,
                   levels=COVERAGE, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    eligible = universe(bars, horizon_bars) if eligible is None else eligible
    n_uni = len(eligible)
    rows = []
    for c in levels:
        sel = select(bars, score, c, mode, horizon_bars, eligible)
        no = non_overlapping(sel)
        rows.append({
            "coverage": c,
            "realised_coverage": len(sel) / n_uni if n_uni else np.nan,
            **summarise(sel),
            "nonoverlap_trades": len(no),
            "nonoverlap_mean_net": float(no["net"].mean()) if len(no) else np.nan,
        })
    return pd.DataFrame(rows)


def breakdown(bars: pd.DataFrame, by) -> pd.DataFrame:
    rows = [{**(dict(zip(by, key)) if isinstance(key, tuple) else {by[0]: key}), **summarise(g)}
            for key, g in bars.groupby(by, observed=True)]
    return pd.DataFrame(rows)


def ev_deciles(bars: pd.DataFrame) -> pd.DataFrame:
    """Is predicted EV itself calibrated? Realised net by predicted-EV decile."""
    q = pd.qcut(bars["ev"], 10, labels=False, duplicates="drop")
    return breakdown(bars.assign(ev_decile=q), ["ev_decile"])


def agreement(a: pd.DataFrame, b: pd.DataFrame, horizon_bars: int, strong: float = 0.2) -> pd.DataFrame:
    """Bucket model A's trades by what model B says on the same bar.

    Direction is the side of each model's best cell when its best EV is positive,
    otherwise flat. "Strong" is the walk-forward top `strong` fraction by EV, so
    neither model's strength cut-off looks at the future.
    """
    def view(bars, tag):
        top = set(select(bars, "ev", strong, "walk_forward", horizon_bars)["ts"])
        return pd.DataFrame({
            "ts": bars["ts"],
            f"dir_{tag}": np.where(bars["ev"] > 0, bars["side"], "flat"),
            f"strong_{tag}": bars["ts"].isin(top),
        })

    m = a.merge(view(a, "a"), on="ts").merge(view(b, "b"), on="ts")
    m = m[m["dir_a"] != "flat"]
    same = m["dir_a"] == m["dir_b"]
    m["bucket"] = np.select(
        [same & m["strong_a"] & m["strong_b"], same, m["dir_b"] == "flat"],
        ["agree_both_strong", "agree", "b_flat"], "oppose")
    return breakdown(m, ["bucket"])


# --- driver -----------------------------------------------------------------

def build(df: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h = cfg["grid"]["horizon_bars"]
    df = calibrate(df, h)
    df = timeout_prior(df, labels, h)
    df = add_ev(df, cfg["costs"])
    return per_bar(df, labels, cfg["costs"], h)


# --- report -----------------------------------------------------------------

RET_COLS = ("mean_pred_ev", "mean_net", "ci_lo", "ci_hi", "nonoverlap_mean_net")


def _bps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in RET_COLS:
        if c in out:
            out[c] = (out[c] * 1e4).round(1)
    for c in ("win_rate", "timeout_rate", "realised_coverage"):
        if c in out:
            out[c] = out[c].round(3)
    if "ess" in out:
        out["ess"] = out["ess"].round(0)
    return out


def calibration_by(long: pd.DataFrame, by: str) -> pd.DataFrame:
    """Predicted vs observed P(TP first), raw and calibrated, per group."""
    rows = []
    for key, g in long.groupby(by, observed=True):
        obs = float((g["outcome"] == lab.Outcome.TP_FIRST).mean())
        rows.append({by: key, "n": len(g), "observed_tp": round(obs, 4),
                     "raw_pred": round(float(g["p_tp_first"].mean()), 4),
                     "cal_pred": round(float(g["c_tp"].mean()), 4)})
    return pd.DataFrame(rows)


def run_report(data_dir: str | Path, cfg: dict, model: str = "lgbm",
               compare: str | None = "jev",
               out: str | Path = "docs/selective_report.md") -> Path:
    from .audit import _md_table
    from .sample import strata

    data_dir = Path(data_dir)
    h = cfg["grid"]["horizon_bars"]
    labels = pd.read_parquet(data_dir / "labels.parquet")
    feats = strata(pd.read_parquet(data_dir / "features.parquet"))[["ts", "year", "vol_q", "dir_t"]]

    def prepare(frame):
        long = add_ev(timeout_prior(calibrate(frame, h), labels, h), cfg["costs"])
        return long, per_bar(long, labels, cfg["costs"], h).merge(feats, on="ts", how="left")

    all_models, match_meta = matched_oof(data_dir, h)
    if model not in all_models:
        raise ValueError(f"model must be one of {sorted(all_models)}")
    if compare == model:
        raise ValueError("compare must differ from model")
    oof = all_models[model]
    long_a, a = prepare(oof)
    a.to_parquet(data_dir / f"selective_{model}_bars.parquet", index=False)
    uni = universe(a, h)
    wf10 = select(a, "ev", 0.10, "walk_forward", h, eligible=uni)

    s = [f"# Selective-trading report — `{model}`", "",
         "Protocol fixed before looking at results: coverage levels "
         f"{list(COVERAGE)}; scores = best-cell EV and directional confidence; "
         "gates are parameter-free; every cut-off that could be traded is walk-forward. "
         "Returns are **net of costs, in basis points per trade**. CIs and ESS are "
         "day-clustered. `diagnostic` cut-offs use the whole period and are **not tradeable**.",
         "", f"Matched observation hash: `{match_meta['observation_hash']}`",
         f"Raw/matched rows: {match_meta['raw_n']} / {match_meta['matched_n']:,}; "
         f"excluded: {match_meta['excluded_n']}; reasons: {match_meta['excluded_reasons']}", "",
         f"Universe: {len(uni):,} bars in {uni['fold'].nunique()} folds "
         f"(first folds are consumed building calibration and cut-off history).", ""]

    best_pos = (uni["ev"] > 0).mean()
    s += ["## Best-available EV after costs", "",
          f"- Bars whose best cell has positive predicted net EV: **{best_pos:.1%}**",
          f"- Best-cell predicted EV, bps — median {uni['ev'].median()*1e4:.1f}, "
          f"p90 {uni['ev'].quantile(.9)*1e4:.1f}, p99 {uni['ev'].quantile(.99)*1e4:.1f}",
          f"- Realised net on bars with EV > 0: "
          f"{uni.loc[uni['ev'] > 0, 'net'].mean()*1e4:.1f} bps over {int((uni['ev'] > 0).sum()):,} bars", "",
          "### Is predicted EV itself calibrated? (realised by predicted-EV decile)", "",
          _md_table(_bps(ev_deciles(uni))), ""]

    for score, title in (("ev", "net EV"), ("conf", "directional confidence")):
        for mode in (("walk_forward", "diagnostic") if score == "ev" else ("walk_forward",)):
            s += [f"## Coverage vs performance — ranked by {title}, {mode.replace('_', '-')}", "",
                  _md_table(_bps(coverage_curve(a, score, mode, h, eligible=uni))), ""]

    for label, frame in (("all bars", uni), ("walk-forward top 10% by EV", wf10)):
        s += [f"## Breakdown — {label}", ""]
        for by in (["vol_q"], ["dir_t"], ["vol_q", "dir_t"], ["tp"], ["side"], ["year"]):
            s += [f"### by {' × '.join(by)}", "", _md_table(_bps(breakdown(frame, by))), ""]
        terc = frame.assign(margin_tercile=pd.qcut(frame["margin"], 3, labels=["low", "mid", "high"]))
        s += ["### by margin over the opposite side (descriptive terciles)", "",
              _md_table(_bps(breakdown(terc, ["margin_tercile"]))), ""]

    cal_long = long_a[long_a["ts"].isin(uni["ts"])].merge(feats[["ts", "vol_q"]], on="ts", how="left")
    s += ["## Calibration after walk-forward isotonic", "",
          "### by TP width", "", _md_table(calibration_by(cal_long, "tp")), "",
          "### by volatility regime", "", _md_table(calibration_by(cal_long, "vol_q")), ""]

    others = []
    if compare:
        if compare not in all_models:
            raise ValueError(f"compare must be one of {sorted(all_models)}")
        others.append((compare, prepare(all_models[compare])[1]))
    for name, b in others:
        s += [f"## Agreement — `{model}` trades bucketed by `{name}`", "",
              _md_table(_bps(agreement(a, b, h))), "",
              f"### `{name}` alone, walk-forward coverage by EV", "",
              _md_table(_bps(coverage_curve(b, "ev", "walk_forward", h))), ""]

    path = Path(out)
    path.write_text("\n".join(s) + "\n", encoding="utf-8")
    return path
