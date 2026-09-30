"""Headline numbers of the VLA-rollout experiment and its thesis tables still come out of the pipeline.

Pins results.json written by src/scripts/analyse_vla_rollout_failure_detection.py (invocation:
docs/REPRODUCE.md). Skips when the JSON is absent (gitignored artifacts).
"""

import pytest

SAFE = "vla-rollout-failure-detection/safe_widowx/split0/results.json"


@pytest.fixture(scope="session")
def safe(load_results):
    return load_results(SAFE)


def test_safe_populations(safe):
    d = safe["checkpoints"]["ft5000"]
    assert (d["n_calib"], d["n_test_success"], d["n_test_failure"]) == (72, 72, 288)
    assert d["episode_length_auroc"] == 0.5, "SAFE episodes are all 50 steps"


def test_safe_preregistered_raw_max_is_inconclusive(safe):
    """Rescored with per-step gripper binarisation (look-ahead fix). Pre-registered primary (episode max of the raw score, step 5000): neither channel reaches
    0.70 and disagreement stays above 0.55 -> inconclusive, not refuted."""
    ch = safe["checkpoints"]["ft5000"]["channels"]
    assert ch["ensemble_disagreement"]["raw_max"]["auroc"] == pytest.approx(0.586, abs=0.002)
    assert ch["value_score"]["raw_max"]["auroc"] == pytest.approx(0.495, abs=0.002)
    assert max(c["raw_max"]["auroc"] for k, c in ch.items()
               if k in ("ensemble_disagreement", "value_score")) < 0.70


def test_safe_time_normalised_value_score(safe):
    op = safe["checkpoints"]["ft5000"]["operating_points"]["value_score"]
    assert op["time_normalised_auroc"] == pytest.approx(0.797, abs=0.003)
    assert op["time_normalised_within_task_auroc"] == pytest.approx(0.801, abs=0.003)
    assert op["false_alarm_rate"] == pytest.approx(0.154, abs=0.003)
    assert op["recall"] == pytest.approx(0.564, abs=0.003)
    assert op["detection_time_median"] == pytest.approx(0.76, abs=0.01)
    zs = safe["checkpoints"]["zeroshot"]["operating_points"]["value_score"]
    assert zs["time_normalised_auroc"] == pytest.approx(0.696, abs=0.003)


def test_safe_baselines_near_chance(safe):
    ops = safe["checkpoints"]["ft5000"]["operating_points"]
    assert ops["policy_token_entropy"]["time_normalised_auroc"] == pytest.approx(0.555, abs=0.003)
    assert ops["action_extremeness"]["time_normalised_auroc"] == pytest.approx(0.553, abs=0.003)


def test_safe_critic_fit_and_swap_control(safe):
    assert safe["checkpoints"]["ft5000"]["critic_fit_calib"]["pearson_r"] == pytest.approx(0.811, abs=0.002)
    assert safe["checkpoints"]["zeroshot"]["critic_fit_calib"]["pearson_r"] == pytest.approx(0.593, abs=0.002)
    sw = {c["channel"]: c for c in safe["swap_controls"]["ft5000"]}
    assert sw["ensemble_disagreement"]["paired_auroc"] == pytest.approx(0.646, abs=0.003)
    assert sw["value_score"]["paired_auroc"] == pytest.approx(0.320, abs=0.003)


LIBERO = "vla-rollout-failure-detection/libero_pi0fast/prefix_{}/results.json"


@pytest.fixture(scope="session")
def libero(load_results):
    return {h: load_results(LIBERO.format(h)) for h in ("min_calib", "min_calib_task", "none")}


def test_libero_length_is_a_perfect_detector_and_the_prefix_removes_it(libero):
    assert libero["none"]["checkpoints"]["ft5000"]["episode_length_auroc"] == 1.0
    d = libero["min_calib"]["checkpoints"]["ft5000"]
    assert d["horizon"] == 29 and d["truncated_length_auroc"] == 0.5
    assert (d["n_calib"], d["n_test_success"], d["n_test_failure"]) == (90, 96, 194)


def test_libero_preregistered_prefix_within_task(libero):
    d = libero["min_calib"]["checkpoints"]["ft5000"]
    ch, op = d["channels"], d["operating_points"]
    assert ch["ensemble_disagreement"]["raw_max_within_task"]["auroc"] == pytest.approx(0.616, abs=0.002)
    assert ch["value_score"]["raw_max_within_task"]["auroc"] == pytest.approx(0.426, abs=0.002)
    assert op["ensemble_disagreement"]["time_normalised_within_task_auroc"] == pytest.approx(0.651, abs=0.003)
    assert op["action_extremeness"]["time_normalised_within_task_auroc"] == pytest.approx(0.572, abs=0.003)
    assert d["critic_fit_calib"]["pearson_r"] == pytest.approx(0.898, abs=0.002)


def test_libero_critic_does_not_beat_action_baseline_on_longer_windows(libero):
    for h, ours, base in (("min_calib_task", 0.749, 0.720), ("none", 0.922, 0.906)):
        op = libero[h]["checkpoints"]["ft5000"]["operating_points"]
        assert op["ensemble_disagreement"]["time_normalised_within_task_auroc"] == pytest.approx(ours, abs=0.003)
        assert op["action_extremeness"]["time_normalised_within_task_auroc"] == pytest.approx(base, abs=0.003)


def test_swap_control_episode_mean(libero, safe):
    """Disagreement rises under another task's instruction on both datasets; the value score on
    LIBERO only. The raw-max comparison of -Qbar points the other way -- a start-of-episode
    artifact, pinned so it cannot silently turn into a claim."""
    for res, sig, val, val_raw in ((libero["min_calib"], 0.687, 0.609, 0.205), (safe, 0.677, 0.473, 0.320)):
        sw = {c["channel"]: c for c in res["swap_controls"]["ft5000"]}
        assert sw["ensemble_disagreement"]["mean_paired_auroc"] == pytest.approx(sig, abs=0.003)
        assert sw["value_score"]["mean_paired_auroc"] == pytest.approx(val, abs=0.003)
        assert sw["value_score"]["paired_auroc"] == pytest.approx(val_raw, abs=0.003)


def test_safe_split1_replicates(load_results):
    r = load_results("vla-rollout-failure-detection/safe_widowx/split1/results.json")
    d = r["checkpoints"]["ft5000"]
    assert d["channels"]["ensemble_disagreement"]["raw_max"]["auroc"] == pytest.approx(0.591, abs=0.002)
    op = d["operating_points"]["value_score"]
    assert op["time_normalised_auroc"] == pytest.approx(0.823, abs=0.003)
    assert op["time_normalised_within_task_auroc"] == pytest.approx(0.821, abs=0.003)
    assert op["false_alarm_rate"] == pytest.approx(0.064, abs=0.003)
    assert op["recall"] == pytest.approx(0.535, abs=0.003)


def _sp(res, ck, key, alpha="0.15"):
    sp = res["checkpoints"][ck]["safe_protocol"][key]
    return sp["roc_auc"], sp["by_alpha"][alpha]


def test_safe_protocol_headline(safe):
    roc, m = _sp(safe, "ft5000", "value_score/step")
    assert m["bal_acc"] == pytest.approx(0.714, abs=0.003)
    assert m["t_det"] == pytest.approx(0.793, abs=0.003)
    roc_c, _ = _sp(safe, "ft5000", "value_score/cumsum")
    assert roc_c == pytest.approx(0.793, abs=0.003)
    roc_ent, m_ent = _sp(safe, "ft5000", "policy_token_entropy/step")
    assert roc_ent == pytest.approx(0.492, abs=0.003)


def test_gripper_heuristic_matches_the_critic_on_safe(safe):
    """The gripper-count baseline: pinned so the comparison cannot silently drift."""
    roc, m = _sp(safe, "ft5000", "gripper_open_steps/step")
    assert roc == pytest.approx(0.853, abs=0.003)
    assert m["bal_acc"] == pytest.approx(0.684, abs=0.003)
    assert safe["checkpoints"]["ft5000"]["channels"]["gripper_open_steps"]["raw_max_within_task"]["auroc"] == \
        pytest.approx(0.885, abs=0.003)


def test_unseen_tasks(load_results):
    u = load_results("vla-rollout-failure-detection/safe_widowx/heldout_tasks_unseen/results.json")
    d = u["checkpoints"]["ft5000"]
    assert (d["n_test_success"], d["n_test_failure"]) == (61, 89)
    assert d["operating_points"]["value_score"]["time_normalised_auroc"] == pytest.approx(0.833, abs=0.003)
    roc_c, _ = _sp(u, "ft5000", "value_score/cumsum")
    assert roc_c == pytest.approx(0.863, abs=0.003)
    _, m_sig = _sp(u, "ft5000", "ensemble_disagreement/step")
    assert m_sig["tnr"] < 0.2, "disagreement should flag unseen-task successes as novel"


def test_libero_continuation(libero):
    d = libero["min_calib"]["checkpoints"]["ft15000"]
    assert d["channels"]["ensemble_disagreement"]["raw_max_within_task"]["auroc"] == pytest.approx(0.694, abs=0.003)
    roc, _ = _sp(libero["none"], "ft15000", "ensemble_disagreement/step")
    assert roc == pytest.approx(0.882, abs=0.003)


@pytest.mark.parametrize("draw, within, bal_ours, bal_grip, recall", [
    ("B", 0.929, 0.568, 0.629, 0.071), ("C", 0.841, 0.546, 0.654, 0.067)])
def test_prespecified_unseen_draws(load_results, draw, within, bal_ours, bal_grip, recall):
    """Draws B and C, criteria fixed before they ran: ranking holds (>= 0.70),
    the seen-task threshold does not transfer (recall < 0.1 at alpha 0.1) and the gripper
    heuristic wins on balanced accuracy."""
    d = load_results(f"vla-rollout-failure-detection/safe_widowx/heldout_tasks_{draw}_unseen/results.json")["checkpoints"]["ft5000"]
    op = d["operating_points"]["value_score"]
    assert op["time_normalised_within_task_auroc"] == pytest.approx(within, abs=0.003)
    assert op["recall"] == pytest.approx(recall, abs=0.003)
    assert d["safe_protocol"]["value_score/step"]["by_alpha"]["0.15"]["bal_acc"] == pytest.approx(bal_ours, abs=0.003)
    assert d["safe_protocol"]["gripper_open_steps/step"]["by_alpha"]["0.15"]["bal_acc"] == pytest.approx(bal_grip, abs=0.003)


# ---- SAFE protocol, 3 folds (thesis tables tab:vla-safe-detection and
# tab:vla-libero-detection, src/scripts/vla_rollout_thesis_tables.py). The asserts above pin the
# earlier single-split analysis (VLA figures of render_thesis_figures.py).
SP = "vla-rollout-failure-detection/safe_protocol/{ds}/fold{k}/{view}/results.json"
SP_VIEWS = {"seen": "B_holdout-cal_evalseen-test", "unseen": "A_evalseen-cal_unseen-test"}


@pytest.mark.parametrize("ds, test_set, ch, mean, std", [
    ("safe_widowx", "seen", "value_score", 0.741, 0.024),
    ("safe_widowx", "seen", "gripper_open_steps", 0.683, 0.020),
    ("safe_widowx", "unseen", "value_score", 0.739, 0.022),
    ("safe_widowx", "unseen", "gripper_open_steps", 0.662, 0.041),
    ("libero_pi0fast", "seen", "value_score", 0.806, 0.064),
    ("libero_pi0fast", "seen", "gripper_open_steps", 0.743, 0.054),
    ("libero_pi0fast", "unseen", "value_score", 0.591, 0.221),
    ("libero_pi0fast", "unseen", "gripper_open_steps", 0.688, 0.100),
])
def test_safe_protocol_rerun_bal_acc(load_results, ds, test_set, ch, mean, std):
    """Balanced accuracy at alpha 0.15, step 50k, all 10 calibration draws: mean +- std (ddof 1)
    over the 3 folds, as in the thesis tables."""
    import statistics
    v = [load_results(SP.format(ds=ds, k=k, view=SP_VIEWS[test_set]))["checkpoints"]["step50000"]
         ["safe_protocol"][f"{ch}/step"]["by_alpha"]["0.15"]["bal_acc"] for k in range(3)]
    assert statistics.fmean(v) == pytest.approx(mean, abs=0.001)
    assert statistics.stdev(v) == pytest.approx(std, abs=0.001)


def test_safe_protocol_rerun_preregistered_criterion(load_results):
    """Fixed before the run: unseen tasks, view A, step 50k, -Qbar within-task
    AUROC >= 0.70 in every fold, and balanced accuracy above the gripper count in every fold."""
    for k, auroc in enumerate((0.864, 0.897, 0.785)):
        d = load_results(SP.format(ds="safe_widowx", k=k, view=SP_VIEWS["unseen"]))["checkpoints"]["step50000"]
        assert d["operating_points"]["value_score"]["time_normalised_within_task_auroc"] == pytest.approx(auroc, abs=0.001)
        sp = d["safe_protocol"]
        assert sp["value_score/step"]["by_alpha"]["0.15"]["bal_acc"] > sp["gripper_open_steps/step"]["by_alpha"]["0.15"]["bal_acc"]
