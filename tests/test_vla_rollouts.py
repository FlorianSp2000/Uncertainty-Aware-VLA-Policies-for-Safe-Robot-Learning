"""Unit tests for say_no.vla_rollouts on synthetic episodes (no dump needed)."""

import numpy as np
import pytest

from say_no import vla_rollouts as vr


def _rec(key, pool, success, task, s, q):
    s, q = np.asarray(s, np.float32), np.asarray(q, np.float32)
    return dict(key=key, pool=pool, success=success, condition=task, length=len(s), s_t=s,
                q_mean_t=q, source=key)


def test_discounted_step_count_matches_mc_return():
    y = vr.discounted_step_count(6, 0.5, 2)
    # rewards -1,-1,-1,-1,0,0 with the last two terminal
    assert y.tolist() == pytest.approx([-1.875, -1.75, -1.5, -1.0, 0.0, 0.0])


def test_truncate_uses_shortest_calibration_success_only():
    recs = [_rec("c1", "calib", True, "a", np.ones(5), -np.ones(5)),
            _rec("c2", "calib", True, "a", np.ones(8), -np.ones(8)),
            _rec("t1", "test", False, "a", np.ones(3), -np.ones(3))]
    out, h = vr.truncate(recs, "min_calib")
    assert h == 5
    assert [r["length"] for r in out] == [5, 5, 3]
    assert recs[1]["length"] == 8, "truncate must not mutate its input"


def test_within_task_auroc_ignores_between_task_level():
    # task a: level 0, task b: level 10. Within each task failures and successes tie, but task b
    # holds most failures, so the pooled AUROC is high while the within-task one is 0.5.
    succ = [_rec(f"s{i}", "test", True, t, [lv], [0]) for i, (t, lv) in
            enumerate([("a", 0), ("a", 0), ("b", 10)])]
    fail = [_rec(f"f{i}", "test", False, t, [lv], [0]) for i, (t, lv) in
            enumerate([("a", 0), ("b", 10), ("b", 10), ("b", 10)])]
    score = lambda r: float(r["s_t"].max())  # noqa: E731
    pooled = vr.auroc_with_std([score(r) for r in succ], [score(r) for r in fail], n_boot=20)
    within = vr.within_task_auroc(succ, fail, score, n_boot=20)
    assert pooled["auroc"] > 0.7
    assert within["auroc"] == pytest.approx(0.5)


def test_operating_point_detects_a_late_rise_and_respects_alpha():
    rng = np.random.default_rng(0)
    T = 30
    ramp = np.linspace(0, 1, T)

    def ep(i, pool, success):
        noise = rng.normal(0, 0.1, T)
        s = noise + (0 if success else 2 * ramp)
        return _rec(f"{pool}{i}{success}", pool, success, "a", s, -s)

    calib = [ep(i, "calib", True) for i in range(200)]
    succ = [ep(i, "test", True) for i in range(200)]
    fail = [ep(i, "test", False) for i in range(100)]
    op = vr.operating_point(calib, succ, fail, ("ensemble_disagreement",), alpha=0.1,
                            split_frac=0.5, n_split_seeds=3)
    assert op["false_alarm_rate"] < 0.2
    assert op["recall"] > 0.9
    # noise sd 0.1 against a ramp of 2: the z-score crosses within the first fifth
    assert 0 < op["detection_time_median"] < 0.2


def test_select_view_calibrates_on_successes_of_one_pool_only():
    dump = dict(records=[_rec("h1", "holdout", True, "a", [1], [-1]),
                         _rec("e1", "eval_seen", True, "a", [1], [-1]),
                         _rec("e2", "eval_seen", False, "a", [1], [-1]),
                         _rec("u1", "unseen", False, "b", [1], [-1])])
    view = vr.select_view(dump, "eval_seen", "unseen")
    assert {(r["key"], r["pool"]) for r in view["records"]} == {("e1", "calib"), ("u1", "test")}
    with pytest.raises(ValueError):
        vr.select_view(dump, "unseen", "unseen")


def test_truncate_per_task_falls_back_to_global_calibration_minimum():
    recs = [_rec("c1", "calib", True, "a", np.ones(5), -np.ones(5)),
            _rec("c2", "calib", True, "b", np.ones(7), -np.ones(7)),
            _rec("t1", "test", False, "b", np.ones(9), -np.ones(9)),
            _rec("t2", "test", False, "unseen_task", np.ones(9), -np.ones(9))]
    out, per_task = vr.truncate(recs, "calib_task_or_global")
    assert per_task == {"a": 5, "b": 7, "unseen_task": 5}
    assert [r["length"] for r in out] == [5, 7, 7, 5]


def test_safe_cp_never_alarms_after_a_truncated_episode_ends():
    rng = np.random.default_rng(0)
    # calibration successes run 10 steps; the score falls after step 5, so does the band
    profile = np.r_[np.full(5, 10.0), np.full(5, 1.0)]
    calib = [_rec(f"c{i}", "calib", True, "a", profile + rng.normal(0, 0.1, 10), -np.ones(10))
             for i in range(20)]
    # a test success observed for 5 steps only, ending at the high early level
    short = _rec("s", "test", True, "a", np.full(5, 10.0), -np.ones(5))
    fail = _rec("f", "test", False, "a", np.full(10, 20.0), -np.ones(10))
    r = vr.safe_functional_cp(calib, [short], [fail], vr.CHANNELS["ensemble_disagreement"], 0.15, 0)
    assert r["tnr"] == 1.0, "padded tail of the short episode must not alarm"
    assert r["tpr"] == 1.0
