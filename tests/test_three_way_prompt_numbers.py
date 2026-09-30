"""Pins the IVA three-way-prompt and episode-operating-point numbers, including the refutations.

A headline that is only written down in prose drifts silently when the analysis is edited. Every case here
asserts over the analysis JSON, and skips loudly when it is absent -- the JSONs are
gitignored artifacts, so this only runs on a machine that produced them.

Reproduce the inputs with:
  src/scripts/cp/analyse_three_way_prompt.py --dump <label>=<pkl> ...
    --out src/results/three-way-prompt-eval-iva/aurocs_and_operating_points_4checkpoints.json
"""

import pytest

TOL = 0.002

REFERENCE = "reference · sr -1 · negdemo 0.10 · ens8 · 50k"
NO_NEGDEMO = "no-negdemo · sr -1 · ens8 · 50k"
WITH_STEP_REWARD = "with_step_reward · sr -1 · negdemo 0.10 · ens4 · 200k"
WITHOUT_STEP_REWARD = "without_step_reward · sr 0 · negdemo 0.10 · ens4 · 200k"


@pytest.fixture(scope="module")
def three_way(load_results):
    return load_results("three-way-prompt-eval-iva/aurocs_and_operating_points_4checkpoints.json")


def test_the_construction_reproduces_the_published_per_task_aurocs(three_way):
    """Frame-ranking check. If this drifts, the evaluation stopped measuring the
    published quantity and none of its comparisons mean anything."""
    published = {
        "sweep_to_dustpan_of_size": 0.943, "open_drawer": 0.885, "close_jar": 0.842,
        "reach_and_drag": 0.774, "meat_off_grill": 0.768, "put_money_in_safe": 0.741,
        "slide_block_to_color_target": 0.736, "turn_tap": 0.648, "push_buttons": 0.546,
    }
    per_task = three_way[REFERENCE]["per_task"]
    for task, expected in published.items():
        assert per_task[task]["auroc_inj"] == pytest.approx(expected, abs=TOL), task


@pytest.mark.parametrize("label,auroc_inj,auroc_other", [
    (REFERENCE, 0.765, 0.789),
    (NO_NEGDEMO, 0.701, 0.770),
    (WITH_STEP_REWARD, 0.885, 0.181),
    (WITHOUT_STEP_REWARD, 0.915, 0.916),
])
def test_three_way_headline(three_way, label, auroc_inj, auroc_other):
    s = three_way[label]["summary"]
    assert s["mean_auroc_inj"] == pytest.approx(auroc_inj, abs=TOL)
    assert s["mean_auroc_other_inj"] == pytest.approx(auroc_other, abs=TOL)


def test_the_fully_trained_checkpoints_show_no_novelty_shortcut(three_way):
    """The refutation. Both 50k checkpoints detect a swapped task prompt at least as
    well as a novel one."""
    for label in (REFERENCE, NO_NEGDEMO):
        assert three_way[label]["summary"]["mean_auroc_drop"] < 0, label


def test_the_two_arms_move_in_opposite_directions_with_training(three_way):
    """At convergence the WITH_STEP_REWARD arm's novelty dependence is enormous and its hard-negative
    AUROC is far BELOW CHANCE, while the WITHOUT_STEP_REWARD arm's has dissolved to zero -- it detects
    a swapped task prompt exactly as well as a novel string. At 10k training steps the
    ranking looked inverted; matched training withdrew that."""
    assert three_way[WITH_STEP_REWARD]["summary"]["mean_auroc_drop"] == pytest.approx(0.704, abs=TOL)
    assert three_way[WITH_STEP_REWARD]["summary"]["mean_auroc_other_inj"] < 0.25
    assert abs(three_way[WITHOUT_STEP_REWARD]["summary"]["mean_auroc_drop"]) < 0.01


def test_at_matched_training_the_without_step_reward_arm_wins_on_both_stimuli(three_way):
    """The withdrawn inversion, pinned in its corrected form."""
    ref, a1 = three_way[REFERENCE]["summary"], three_way[WITHOUT_STEP_REWARD]["summary"]
    assert a1["mean_auroc_inj"] > ref["mean_auroc_inj"]
    assert a1["mean_auroc_other_inj"] > ref["mean_auroc_other_inj"]


def _op(model, frame_statistic, score, statistic="max", alpha=0.10, pool_tasks=False):
    """The per-task-calibrated row by default. The pooled row is a different detector --
    same score, larger calibration set -- so it must be selected explicitly, never matched
    by accident."""
    hits = [o for o in model["operating_points"]
            if o["frame_statistic"] == frame_statistic and o["score"] == score
            and o["statistic"] == statistic and o["alpha"] == alpha
            and o.get("pool_tasks", False) == pool_tasks]
    assert len(hits) == 1, f"{frame_statistic}/{score}/{statistic} not uniquely present"
    return hits[0]


def test_short_truncated_episodes_stay_out_of_calibration(three_way):
    """The fp split ends a terminal-refusal episode AT the refusal. Under align="truncate"
    one 2-frame episode collapses mu_t/varsigma(t) to length 2 and the z-score stops
    conditioning on time. 11 of 224 episodes are excluded from calibration for that reason;
    all 224 are still scored."""
    o = _op(three_way[WITHOUT_STEP_REWARD], "both", "raw")
    assert o["align"] == "truncate"
    assert o["min_calib_len"] == 20
    assert o["n_episodes_excluded_from_calibration"] == 11
    assert o["n_feasible_episodes"] == 224


def test_the_operating_point_that_meets_the_target(three_way):
    """FAR <= 0.10 at recall >= 0.70 on BOTH stimulus types."""
    o = _op(three_way[WITHOUT_STEP_REWARD], "both", "raw")
    assert o["episode_far"] == pytest.approx(0.100, abs=0.01)
    assert o["episode_recall_iva_injection"] == pytest.approx(0.804, abs=0.01)
    assert o["episode_recall_foreign_instruction"] == pytest.approx(0.904, abs=0.01)
    assert o["episode_far"] <= 0.10
    assert o["recall_iva_ci"][0] >= 0.70
    assert o["recall_foreign_ci"][0] >= 0.70


def test_the_project_score_alone_misses_the_deployment_stimulus(three_way):
    """std_Q at a comparable false-alarm rate: looks fine on IVA, far worse on a real wrong
    instruction. This is the single number the whole control exists to expose. On the WITH_STEP_REWARD arm
    the same split is 0.575 against 0.035 -- a detector reading std_Q there would miss 96.5 % of
    real wrong-instruction episodes while its benchmark AUROC read 0.885."""
    o = _op(three_way[WITHOUT_STEP_REWARD], "disagreement", "raw")
    assert o["episode_recall_iva_injection"] > 0.70
    assert o["episode_recall_foreign_instruction"] == pytest.approx(0.548, abs=0.01)


def test_which_channel_works_depends_on_what_the_model_learned(three_way):
    """Not "mean Q needs the step-reward horizon gone" -- that was the 10k reading. At 50k mean Q works on
    BOTH newly trained arms (0.962 step-reward horizon, 0.998 step-reward horizon-free) because the negative-demo mapping
    drives Q towards the ~-51 fixed point either way. What fails is DISAGREEMENT on the WITH_STEP_REWARD arm
    (0.121). The "-50 floor is an absorbing state where members agree" explanation once given here is
    REFUTED (soft attractor, and a Simpson effect -- at matched mean Q the swapped-prompt
    spread is higher); the numbers stay pinned, the mechanism is withdrawn."""
    a0_q = _op(three_way[WITH_STEP_REWARD], "neg_mean_q", "raw")["episode_recall_foreign_instruction"]
    a1_q = _op(three_way[WITHOUT_STEP_REWARD], "neg_mean_q", "raw")["episode_recall_foreign_instruction"]
    a0_s = _op(three_way[WITH_STEP_REWARD], "disagreement", "raw")["episode_recall_foreign_instruction"]
    assert a0_q > 0.95 and a1_q > 0.90
    assert a0_s < 0.10   # 0.035 at 200k
    # The reference is the mirror image: it saturates on only 36.6 % of frames, so its
    # disagreement still carries signal while its mean Q does not.
    ref = three_way[REFERENCE]
    assert _op(ref, "disagreement", "raw")["episode_recall_foreign_instruction"] > 0.60
    assert _op(ref, "neg_mean_q", "raw")["episode_recall_foreign_instruction"] < 0.30


def test_combining_channels_is_not_free(three_way):
    """Calibrating the max over two channels raises the threshold, so a channel that carries
    no signal costs recall. On WITH_STEP_REWARD that is 0.962 -> 0.410; on WITHOUT_STEP_REWARD, where both channels are
    informative, it is a wash. Corrects an earlier claim that the hedge costs nothing."""
    a0 = three_way[WITH_STEP_REWARD]
    assert _op(a0, "both", "raw")["episode_recall_foreign_instruction"] < 0.5
    assert _op(a0, "neg_mean_q", "raw")["episode_recall_foreign_instruction"] > 0.95
    a1 = three_way[WITHOUT_STEP_REWARD]
    assert abs(_op(a1, "both", "raw")["episode_recall_foreign_instruction"]
               - _op(a1, "neg_mean_q", "raw")["episode_recall_foreign_instruction"]) < 0.03


def test_no_task_fails_on_the_deployment_stimulus(three_way):
    """A 9-task mean must not hide a failing task.

    On the deployment-relevant stimulus none does: 1.000 on eight of nine tasks, 0.922 on the
    ninth. On IVA's injections exactly one task fails badly -- slide_block_to_color_target at
    0.203, which got WORSE with training (0.435 at 10k) and has no explanation yet. Reported,
    not tuned away: 25 eval episodes per task and no third split, so per-task tuning would fit
    the numbers being reported.
    """
    per_task = _op(three_way[WITHOUT_STEP_REWARD], "both", "raw")["per_task"]
    # Eight of nine tasks are at 1.000 on the deployment stimulus; push_buttons collapses to
    # 0.137 at 200k having been at 1.000 at 50k, and that collapse is NOT covered by the known
    # push_buttons instruction artefact, which is about its injections.
    assert sorted(t for t, v in per_task.items() if v["recall_foreign"] < 0.9) == ["push_buttons"]
    weak = sorted(t for t, v in per_task.items() if v["recall_iva"] < 0.70)
    assert weak == ["push_buttons", "slide_block_to_color_target"]


@pytest.fixture(scope="module")
def pooled(load_results):
    """Cross-task pooled sweep. Produce with:
      analyse_three_way_prompt.py --dump "without_step_reward 50k"=<pkl> --alphas 0.10,0.05,0.03
                         --pool-tasks --out src/results/three-way-prompt-eval-iva/alpha_sweep_without_step_reward_200k_pooled.json
    """
    return load_results("three-way-prompt-eval-iva/alpha_sweep_without_step_reward_200k_pooled.json")["without_step_reward @200k"]


def _pooled_op(model, alpha):
    hits = [o for o in model["operating_points"]
            if o["pool_tasks"] and o["frame_statistic"] == "both" and o["score"] == "raw"
            and o["statistic"] == "max" and o["alpha"] == alpha]
    assert len(hits) == 1, f"alpha={alpha} pooled row not uniquely present"
    return hits[0]


def test_the_realised_rate_tracks_the_level_you_ask_for(pooled):
    """"Operable at a fixed threshold" means the dial works, not that one point happens to
    land. Cross-task pooling is what makes levels below 1/11 attainable at all."""
    for alpha, expected in [(0.10, 0.077), (0.05, 0.046), (0.03, 0.022)]:
        assert _pooled_op(pooled, alpha)["episode_far"] == pytest.approx(expected, abs=0.015)


def test_the_headline_operating_point_clears_the_target_on_intervals(pooled):
    """The strongest form of the claim: the whole false-alarm INTERVAL under 0.10, not just
    the point estimate, with both recalls above 0.70."""
    o = _pooled_op(pooled, 0.05)
    assert o["far_ci"][1] <= 0.10, o["far_ci"]
    assert o["episode_recall_iva_injection"] >= 0.70
    assert o["episode_recall_foreign_instruction"] >= 0.70
    assert o["recall_foreign_ci"][0] >= 0.70


def test_c3_wrong_scene_interval_is_on_distinct_episodes(pooled):
    """The 8 swapped prompts of one episode alarm together (ICC ~ 1), so the
    interval is on the 224 distinct episodes, not the 1792 scorings. Was [0.871, 0.901]."""
    o = _pooled_op(pooled, 0.05)
    assert o["n_foreign_episode_scorings"] == 1792
    assert o["n_eff_foreign"] == 224
    assert o["recall_foreign_ci"] == pytest.approx([0.839, 0.922], abs=TOL)


def test_c4_there_are_189_false_premise_episodes_not_144(pooled):
    o = _pooled_op(pooled, 0.05)
    assert o["n_iva_episodes"] == 189
    assert o["recall_iva_ci"] == pytest.approx([0.683, 0.806], abs=TOL)


def test_c6_mean_q_alone_is_at_least_as_good_as_both_channels(pooled):
    both = _pooled_op(pooled, 0.05)
    mq = [o for o in pooled["operating_points"]
          if o["pool_tasks"] and o["frame_statistic"] == "neg_mean_q" and o["score"] == "raw"
          and o["statistic"] == "max" and o["alpha"] == 0.05]
    assert len(mq) == 1
    mq = mq[0]
    assert mq["episode_far"] == pytest.approx(0.046, abs=TOL) == both["episode_far"]
    assert mq["episode_recall_iva_injection"] == pytest.approx(0.767, abs=TOL)
    assert mq["episode_recall_foreign_instruction"] == pytest.approx(0.888, abs=TOL)
    assert both["episode_recall_iva_injection"] == pytest.approx(0.750, abs=TOL)
    assert both["episode_recall_foreign_instruction"] == pytest.approx(0.887, abs=TOL)
