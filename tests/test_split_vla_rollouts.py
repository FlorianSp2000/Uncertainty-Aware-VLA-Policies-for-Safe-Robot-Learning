"""SAFE-protocol splits of the VLA-rollout caches (external/V-GPS/experiments/split_vla_rollouts.py)."""
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).parents[1] / "external/V-GPS/experiments/split_vla_rollouts.py"
_spec = importlib.util.spec_from_file_location("split_vla_rollouts", _path)
split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split)

TASKS = [f"t{i}" for i in range(8)]


def test_folds_hold_out_disjoint_task_blocks():
    folds = [split.unseen_tasks(TASKS, 0, k, 2) for k in range(3)]
    flat = [t for f in folds for t in f]
    assert len(flat) == len(set(flat)) == 6
    with pytest.raises(ValueError):
        split.unseen_tasks(TASKS, 0, 4, 2)


def test_pools_follow_safe_protocol():
    rows = [dict(key=f"{t}_{o}_{i}", condition=t, success=o)
            for t in TASKS for o in (True, False) for i in range(12)]
    unseen = ["t0", "t1"]
    split.assign_pools(rows, unseen, seen_eval_frac=1 / 3, holdout_frac=0.25, split_seed=0)
    for r in rows:
        if r["condition"] in unseen:
            assert r["pool"] == "unseen"
        elif not r["success"]:
            assert r["pool"] in ("eval_seen", "train_fail")
        else:
            assert r["pool"] in ("eval_seen", "holdout", "finetune")
    per = lambda pool, o: sum(r["pool"] == pool and r["success"] == o and r["condition"] == "t2" for r in rows)  # noqa: E731
    assert (per("eval_seen", True), per("holdout", True), per("finetune", True)) == (4, 2, 6)
    assert (per("eval_seen", False), per("train_fail", False)) == (4, 8)


def test_foresight_cv_folds_partition_episodes_and_split_rest_6_1_1():
    rows = [dict(key=f"{t}_{o}_{i}", condition=t, success=o)
            for t in TASKS[:2] for o in (True, False) for i in range(24)]
    tests = []
    for fold in range(3):
        rs = [dict(r) for r in rows]
        split.assign_pools_cv(rs, fold, split_seed=0)
        tests.append({r["key"] for r in rs if r["pool"] == "cv_test"})
        succ = [r for r in rs if r["condition"] == "t0" and r["success"] and r["pool"] != "cv_test"]
        assert [sum(r["pool"] == p for r in succ) for p in ("finetune", "holdout", "cv_calib")] == [12, 2, 2]
        assert all(r["pool"] in ("cv_test", "train_fail") for r in rs if not r["success"])
    assert set.union(*tests) == {r["key"] for r in rows} and sum(map(len, tests)) == len(rows)
