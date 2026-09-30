"""策略政策保护测试。 / Strategy-policy guards.

合成赢家诅咒测试是实验对照：真实 EV 相同时，argmax 应出现选择偏差，
固定单元则不应如此；否则真实数据评估结果不可信。

The synthetic winner's-curse test is the control for the whole experiment:
when every cell's true EV is identical and predictions are unbiased noise,
hard argmax must show a large positive selection bias and a fixed cell must
show none. If the evaluator could not see that, its verdict on real data
would mean nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jev_trader.policy import (
    ArgmaxEVPolicy, Cubes, FixedCellPolicy, LocalRobustPolicy, MultiTPPolicy,
    SoftmaxEVPolicy, Surface, evaluate, metrics, to_cubes,
    calibrate_ev,
)
from jev_trader.selective import select
from jev_trader.selective import universe

GRID = {"tp": [0.005, 0.010, 0.015, 0.020, 0.030], "sl": [0.005, 0.0075, 0.010, 0.015],
        "horizon_bars": 16}
POLICIES = [FixedCellPolicy(GRID, 0.015, 0.010), ArgmaxEVPolicy(), LocalRobustPolicy(1.0),
            SoftmaxEVPolicy(5e-4), SoftmaxEVPolicy(100e-4), MultiTPPolicy(20e-4)]


def random_ev(n=500, seed=0):
    return np.random.default_rng(seed).normal(0, 0.002, (n, 2, 5, 4))


@pytest.mark.parametrize("pol", POLICIES, ids=lambda p: p.name)
def test_weights_are_a_one_sided_allocation(pol):
    w = pol.weights(random_ev())
    assert (w >= 0).all()
    assert np.allclose(w.sum((1, 2, 3)), 1.0)
    per_side = w.sum((2, 3))
    assert ((per_side[:, 0] == 0) | (per_side[:, 1] == 0)).all(), "never long and short at once"


def test_softmax_limits_are_argmax_and_uniform():
    ev = random_ev()
    cold = SoftmaxEVPolicy(1e-9).weights(ev)
    assert np.allclose(cold, ArgmaxEVPolicy().weights(ev))
    hot = SoftmaxEVPolicy(1e3).weights(ev)
    side_mass = hot[hot > 0]
    assert np.allclose(side_mass, 1 / 20, atol=1e-6)


def test_multi_tp_shares_one_stop():
    ev = random_ev(n=50)
    for i in range(50):
        plan = MultiTPPolicy(20e-4).build_plan(Surface(ev[i], tuple(GRID["tp"]), tuple(GRID["sl"]), 16))
        assert plan.stop is not None and all(leg.sl == plan.stop for leg in plan.legs)
        assert len(plan.legs) == 5
        assert sum(leg.weight for leg in plan.legs) == pytest.approx(1.0)


def test_fixed_cell_only_chooses_the_side():
    ev = random_ev()
    w = FixedCellPolicy(GRID, 0.015, 0.010).weights(ev)
    assert (w[:, :, 2, 2].sum(1) == 1).all()
    assert w.sum() == pytest.approx(len(ev))


def test_local_robust_prefers_a_supported_plateau_over_a_spike():
    """A 面是平台，B 面是孤立尖峰；稳健政策应选择平台而非 argmax 尖峰。

    Surface A: 13, 14, 13, 12 bps in one region. Surface B: -4, -2, +14.5, -5.
    Argmax takes the spike; the robust policy must take the plateau."""
    ev = np.full((2, 5, 4), -0.0010)
    ev[0, 1:3, 1:3] = np.array([[13, 14], [13, 12]]) * 1e-4        # 多头平台 / Long plateau.
    ev[1, 2, 0:4] = np.array([-4, -2, 14.5, -5]) * 1e-4            # 空头孤立尖峰 / Short isolated spike.
    s = Surface(ev, tuple(GRID["tp"]), tuple(GRID["sl"]), 16)
    assert ArgmaxEVPolicy().build_plan(s).side == "short"
    assert LocalRobustPolicy(1.0).build_plan(s).side == "long"


def test_plan_matches_the_vectorised_core():
    ev = random_ev(n=1)
    s = Surface(ev[0], tuple(GRID["tp"]), tuple(GRID["sl"]), 16)
    for pol in POLICIES:
        plan = pol.build_plan(s)
        assert plan.predicted_ev == pytest.approx(float((pol.weights(ev) * ev).sum()))
        assert sum(leg.weight for leg in plan.legs) == pytest.approx(1.0)
        assert plan.entry == "next_open" and plan.timeout_bars == 16


def test_winners_curse_is_visible_to_the_evaluator():
    """各单元真实 EV 均为 0，预测和结果是独立无偏噪声；差距纯属选择偏差。

    True EV of every cell is 0; predictions and realisations are independent
    unbiased noise. Any policy's predicted-minus-realised gap is pure selection."""
    rng = np.random.default_rng(1)
    n = 20_000
    pred = rng.normal(0, 0.002, (n, 2, 5, 4))
    real = rng.normal(0, 0.002, (n, 2, 5, 4))
    ts = np.arange(n) * 900_000
    c = Cubes(ts, np.zeros(n, int), pred, real, real, 0)
    days = np.unique(ts // 86_400_000)
    bias = {p.name: metrics(evaluate(p, c), n, days, 16)["selection_bias_bps"] for p in POLICIES}
    assert bias["argmax EV"] > 30                                  # 40 个正态变量最大值期望 / Expected max of 40 normals.
    assert abs(bias["fixed tp1.50% sl1.00%"]) < bias["argmax EV"] / 3   # 只二选一 / Two-way choice only.
    assert bias["softmax tau=100bps"] < bias["softmax tau=5bps"] < bias["argmax EV"]


def test_to_cubes_places_cells_and_drops_incomplete_bars():
    rows = []
    for ts in (0, 900_000):
        for side in ("long", "short"):
            for tp in GRID["tp"]:
                for sl in GRID["sl"]:
                    if ts == 900_000 and side == "short" and tp == 0.030 and sl == 0.015:
                        continue                                    # 一个歧义单元 / One ambiguous cell.
                    rows.append({"ts": ts, "fold": 0, "side": side, "tp": np.float32(tp),
                                 "sl": np.float32(sl), "ev": tp - sl, "net": 0.0, "gross": 0.0})
    c = to_cubes(pd.DataFrame(rows), GRID)
    assert c.dropped == 1 and len(c.ts) == 1
    assert c.ev[0, 1, 4, 0] == pytest.approx(0.030 - 0.005)


def test_policy_ev_calibration_uses_only_earlier_folds_and_recovers_linear_rule():
    rows = []
    for fold in range(3):
        for i in range(4):
            pred = float(fold * 4 + i)
            rows.append({"ts": fold * 20_000_000 + i, "fold": fold, "pred": pred,
                         "score": pred, "net": 2.0 + 0.3 * pred,
                         "gross": 0.0, "day": fold, "eff_legs": 1.0})
    frame = pd.DataFrame(rows)
    calibrated, report = calibrate_ev(frame)
    assert calibrated.index.tolist() == frame.index.tolist()
    assert set(report.fold) == {1, 2}
    assert report.loc[report.fold == 1, "a"].iloc[0] == pytest.approx(2.0)
    assert report.loc[report.fold == 1, "b"].iloc[0] == pytest.approx(0.3)
    assert calibrated.loc[calibrated.fold == 0, "calibrated_pred"].isna().all()
    test = calibrated[calibrated.calibrated_pred.notna()]
    assert np.array_equal(np.argsort(test.pred), np.argsort(test.calibrated_pred))


def test_policy_ev_calibration_never_reverses_a_negative_slope():
    rows = []
    for fold in range(2):
        for i in range(4):
            pred = float(fold * 4 + i)
            rows.append({"ts": fold * 20_000_000 + i, "fold": fold, "pred": pred,
                         "score": pred, "net": 10.0 - pred,
                         "gross": 0.0, "day": fold, "eff_legs": 1.0})
    calibrated, report = calibrate_ev(pd.DataFrame(rows))
    assert report.loc[report.fold == 1, "b"].iloc[0] == 0
    assert calibrated.loc[calibrated.fold == 1, "calibrated_pred"].nunique() == 1
    assert calibrated.loc[calibrated.fold == 1, "score"].tolist() == [4.0, 5.0, 6.0, 7.0]


def test_policy_ev_calibration_excludes_unresolved_fold_boundary_labels():
    current_start = (16 + 2) * 900_000
    boundary = current_start - 1
    rows = [
        {"ts": 0, "fold": 0, "pred": 0.0, "score": 0.0, "net": 2.0, "gross": 0.0, "day": 0, "eff_legs": 1.0},
        {"ts": 900_000, "fold": 0, "pred": 1.0, "score": 1.0, "net": 3.0, "gross": 0.0, "day": 0, "eff_legs": 1.0},
        {"ts": boundary, "fold": 0, "pred": 100.0, "score": 100.0, "net": -999.0, "gross": 0.0, "day": 0, "eff_legs": 1.0},
        {"ts": current_start, "fold": 1, "pred": 0.5, "score": 0.5, "net": 2.5, "gross": 0.0, "day": 1, "eff_legs": 1.0},
    ]
    clean, _ = calibrate_ev(pd.DataFrame(rows))
    changed = pd.DataFrame(rows)
    changed.loc[2, "net"] = 999.0
    changed_result, _ = calibrate_ev(changed)
    assert clean.loc[3, "calibrated_pred"] == pytest.approx(2.5)
    assert changed_result.loc[3, "calibrated_pred"] == pytest.approx(clean.loc[3, "calibrated_pred"])


def test_policy_calibrates_full_frame_before_final_universe_filter():
    rows = []
    for fold in range(5):
        for i in range(4):
            ts = fold * 20_000_000 + i
            pred = float(fold + i / 10)
            rows.append({"ts": ts, "fold": fold, "pred": pred, "score": pred,
                         "net": 1.0 + 0.2 * pred, "gross": 0.0, "day": fold,
                         "eff_legs": 1.0})
    frame = pd.DataFrame(rows)
    eligible = universe(frame[["ts", "fold"]], 16)
    full_calibrated, _ = calibrate_ev(frame, 16)
    final = full_calibrated[full_calibrated["ts"].isin(eligible["ts"])]
    assert len(final) < len(frame)
    assert len(final) == len(eligible)
    assert final["calibrated_pred"].notna().all()
    filtered_first, _ = calibrate_ev(frame[frame["ts"].isin(eligible["ts"])], 16)
    assert filtered_first.loc[filtered_first["fold"] == 3, "calibrated_pred"].isna().all()


def test_policy_selection_uses_full_history_for_first_eligible_fold():
    rows = []
    for fold in range(5):
        for i in range(4):
            ts = fold * 20_000_000 + i
            rows.append({"ts": ts, "fold": fold, "score": fold + i / 10,
                         "net": 0.0, "gross": 0.0})
    full = pd.DataFrame(rows)
    eligible = universe(full, 16)

    selected = select(full, "score", 0.5, "walk_forward", 16, eligible=eligible)

    assert set(eligible["fold"]) == {3, 4}
    assert (selected["fold"] == 3).any()
