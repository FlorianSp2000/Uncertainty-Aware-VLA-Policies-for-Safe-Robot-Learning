"""Pins every number in the thesis tables of src/results/episode-operating-point-iva/thesis/.

Producer: src/scripts/cp/analyse_episode_statistics.py (exact command in the JSON's "command").
without_step_reward @200k, both channels, alpha = 0.05, tau_alpha = 0.05, cross-task pooled
quantile, 5-fold x analysis seeds 0-2. Skips when the JSON is absent (gitignored artifact).
"""

import pytest

TOL = 0.0015


@pytest.fixture(scope="module")
def res(load_results):
    return load_results("episode-operating-point-iva/thesis/episode_statistics.json")


def _row(res, statistic, frame_statistic="both"):
    hits = [r for r in res["statistics"]
            if r["statistic"] == statistic and r["frame_statistic"] == frame_statistic]
    assert len(hits) == 1, (statistic, frame_statistic)
    return hits[0]


def test_provenance(res):
    assert res["step"] == 200000 and res["n_members"] == 4
    assert res["alpha"] == 0.05 and res["tau_alpha"] == 0.05
    assert res["seeds"] == [0, 1, 2] and res["n_folds"] == 5
    for r in res["statistics"]:
        assert r["pool_tasks"] and r["n_calibrations"] == 15
        assert r["n_feasible_episodes"] == 224 and r["n_iva_episodes"] == 189
        assert r["n_foreign_episode_scorings"] == 1792 and r["n_eff_foreign"] == 224


# statistic, frame channels: FAR, FAR CI, recall IVA, CI, recall wrong scene, CI, AUROC IVA, AUROC wrong scene
TABLE = [
    ("max", "both", .046, (.025, .082), .750, (.683, .806), .887, (.839, .922), .901, .949),
    ("topk_mean5", "both", .048, (.027, .084), .728, (.661, .787), .887, (.839, .922), .883, .955),
    ("count_above", "both", .040, (.021, .075), .049, (.026, .090), .640, (.575, .700), .723, .922),
    ("frac_above", "both", .054, (.031, .091), .063, (.037, .108), .909, (.865, .940), .729, .958),
    ("cusum", "both", .048, (.027, .084), .660, (.589, .723), .885, (.837, .920), .884, .934),
    ("max", "neg_mean_q", .046, (.025, .082), .767, (.702, .822), .888, (.840, .923), .884, .971),
]


@pytest.mark.parametrize("stat,fs,far,far_ci,ri,ri_ci,rf,rf_ci,ai,af", TABLE)
def test_episode_statistics_table(res, stat, fs, far, far_ci, ri, ri_ci, rf, rf_ci, ai, af):
    r = _row(res, stat, fs)
    assert r["episode_far"] == pytest.approx(far, abs=TOL)
    assert r["far_ci"] == pytest.approx(far_ci, abs=TOL)
    assert r["episode_recall_iva_injection"] == pytest.approx(ri, abs=TOL)
    assert r["recall_iva_ci"] == pytest.approx(ri_ci, abs=TOL)
    assert r["episode_recall_foreign_instruction"] == pytest.approx(rf, abs=TOL)
    assert r["recall_foreign_ci"] == pytest.approx(rf_ci, abs=TOL)
    assert r["episode_auroc_iva_injection"] == pytest.approx(ai, abs=TOL)
    assert r["episode_auroc_foreign_instruction"] == pytest.approx(af, abs=TOL)


def test_the_headline_row_is_the_one_already_published(res):
    """The refactor into say_no.cp.episode_eval must reproduce the alpha-sweep headline."""
    r = _row(res, "max")
    assert r["episode_far"] == pytest.approx(31 / 672)


def test_counting_statistics_miss_impulses_but_not_sustained_shifts(res):
    """IVA injections are ~16 one-frame impulses per episode; a count or fraction of
    exceedances cannot tell them from the feasible episodes whose whole trace sits high.
    Holds at every tau in the sweep, so it is not a choice of tau."""
    for r in res["tau_alpha_sweep"]:
        if r["statistic"] in ("count_above", "frac_above"):
            assert r["episode_recall_iva_injection"] < 0.10, r["tau_alpha"]
        assert r["episode_far"] < 0.06
    assert _row(res, "frac_above")["episode_recall_foreign_instruction"] > 0.90


def test_frame_versus_episode(res):
    f = res["frame_level"]
    assert f["n_feasible_frames"] == 24236 and f["n_injected_frames"] == 2925
    assert f["n_foreign_frames"] == 217288
    assert f["frame_auroc_iva_injection"] == pytest.approx(0.890, abs=TOL)
    assert f["frame_auroc_foreign_instruction"] == pytest.approx(0.988, abs=TOL)
    assert f["frame_fpr"] == pytest.approx(0.045, abs=TOL)
    assert f["frame_tpr_iva_injection"] == pytest.approx(0.774, abs=TOL)
    assert f["frame_tpr_foreign_instruction"] == pytest.approx(0.911, abs=TOL)
    # the same frame threshold counted per episode: the false-alarm rate the episode unit fixes
    assert f["episode_far_any_frame"] == pytest.approx(0.202, abs=TOL)
    assert f["episode_recall_iva_any_frame"] == pytest.approx(0.866, abs=TOL)
    assert f["episode_recall_foreign_any_frame"] == pytest.approx(0.889, abs=TOL)


# statistic, frame channels: IVA median, mean, before onset, recall after onset; wrong scene median, mean, at frame 0
DELAYS = [
    ("max", "both", 0, 2.9, .019, .735, 0, 0.3, .985),
    ("count_above", "both", 102, 103.0, .000, .049, 99, 95.7, .000),
    ("cusum", "both", 0, 9.4, .008, .654, 0, 1.0, .810),
    ("max", "neg_mean_q", 0, 1.7, .028, .746, 0, 0.1, .992),
]


def test_single_calibration_far_spread(res):
    """The pooled FAR averages 15 (seed, fold) calibrations; one deployment calibrates once."""
    f = _row(res, "max")["episode_far_by_calibration"]
    assert len(f) == 15
    assert min(f) == pytest.approx(0.000, abs=TOL) and max(f) == pytest.approx(0.156, abs=TOL)


# frame channels, alpha: FAR, FAR CI, recall IVA, CI, recall wrong scene, CI
ALPHA_SWEEP = [
    ("both", .10, .077, (.049, .120), .785, (.721, .837), .888, (.840, .923)),
    ("both", .05, .046, (.025, .082), .750, (.683, .806), .887, (.839, .922)),
    ("both", .03, .022, (.010, .051), .716, (.648, .776), .855, (.803, .895)),
    ("neg_mean_q", .10, .074, (.047, .116), .795, (.732, .847), .888, (.840, .923)),
    ("neg_mean_q", .05, .046, (.025, .082), .767, (.702, .822), .888, (.840, .923)),
    ("neg_mean_q", .03, .022, (.010, .051), .720, (.652, .779), .872, (.822, .910)),
]


def _assert_op(o, far, far_ci, ri, ri_ci, rf, rf_ci):
    assert o["episode_far"] == pytest.approx(far, abs=TOL)
    assert o["far_ci"] == pytest.approx(far_ci, abs=TOL)
    assert o["episode_recall_iva_injection"] == pytest.approx(ri, abs=TOL)
    assert o["recall_iva_ci"] == pytest.approx(ri_ci, abs=TOL)
    assert o["episode_recall_foreign_instruction"] == pytest.approx(rf, abs=TOL)
    assert o["recall_foreign_ci"] == pytest.approx(rf_ci, abs=TOL)


@pytest.mark.parametrize("fs,alpha,far,far_ci,ri,ri_ci,rf,rf_ci", ALPHA_SWEEP)
def test_operating_points_table(res, fs, alpha, far, far_ci, ri, ri_ci, rf, rf_ci):
    hits = [o for o in res["alpha_sweep"] if o["frame_statistic"] == fs and o["alpha"] == alpha]
    assert len(hits) == 1
    assert hits[0]["pool_tasks"] and hits[0]["statistic"] == "max"
    _assert_op(hits[0], far, far_ci, ri, ri_ci, rf, rf_ci)


# training seed: FAR, FAR CI, recall IVA, CI, recall wrong scene, CI -- both at step 50k
REPLICATES = [
    (44, .042, (.022, .076), .862, (.806, .904), .993, (.970, .998)),
    (45, .055, (.032, .093), .825, (.765, .873), .933, (.892, .959)),
]


@pytest.mark.parametrize("seed,far,far_ci,ri,ri_ci,rf,rf_ci", REPLICATES)
def test_training_seed_replicates(res, seed, far, far_ci, ri, ri_ci, rf, rf_ci):
    hits = [o for o in res["replicates"] if o["training_seed"] == seed]
    assert len(hits) == 1
    o = hits[0]
    assert o["step"] == 50000 and o["n_members"] == 4 and o["alpha"] == res["alpha"]
    _assert_op(o, far, far_ci, ri, ri_ci, rf, rf_ci)


# task: episodes, IVA episodes, FAR, recall IVA, recall wrong scene (headline row)
PER_TASK = [
    ("close_jar", 25, 20, .013, .967, 1.000),
    ("meat_off_grill", 25, 21, .080, .889, .987),
    ("open_drawer", 24, 19, .056, .895, 1.000),
    ("push_buttons", 25, 21, .013, .032, .000),
    ("put_money_in_safe", 25, 21, .067, .952, 1.000),
    ("reach_and_drag", 25, 19, .000, .947, 1.000),
    ("slide_block_to_color_target", 25, 23, .080, .174, 1.000),
    ("sweep_to_dustpan_of_size", 25, 22, .000, 1.000, 1.000),
    ("turn_tap", 25, 23, .107, .957, 1.000),
]


@pytest.mark.parametrize("task,n,niva,far,ri,rf", PER_TASK)
def test_per_task_table(res, task, n, niva, far, ri, rf):
    v = _row(res, "max")["per_task"][task]
    assert v["n_feasible_episodes"] == n and v["n_iva_episodes"] == niva
    assert v["far"] == pytest.approx(far, abs=TOL)
    assert v["recall_iva"] == pytest.approx(ri, abs=TOL)
    assert v["recall_foreign"] == pytest.approx(rf, abs=TOL)


def test_eval_split_table(res):
    s = res["eval_split"]
    assert len(s) == 9
    assert sum(v["n_episodes"] for v in s.values()) == 224
    assert sum(v["n_iva_episodes"] for v in s.values()) == 189
    assert sum(v["n_frames"] for v in s.values()) == 27161
    assert sum(v["n_injected_frames"] for v in s.values()) == 2925
    assert s["open_drawer"]["n_episodes"] == 24


# step reward, negdemo ratio, members, step: frame AUROC sigma_Q IVA injection, wrong scene
REWARD_DESIGN = [
    (-1, .10, 7, 50000, .765, .789),
    (-1, .00, 8, 50000, .701, .770),
    (-1, .10, 4, 200000, .885, .181),
    (0, .10, 4, 200000, .915, .916),
]


@pytest.mark.parametrize("sr,nd,nm,step,ai,af", REWARD_DESIGN)
def test_frame_auroc_reward_design_table(res, sr, nd, nm, step, ai, af):
    hits = [r for r in res["reward_design"] if r["step_reward"] == sr and r["step"] == step
            and r["negdemo_ratio"] == nd and r["n_members"] == nm]
    assert len(hits) == 1
    assert hits[0]["auroc_iva_injection"] == pytest.approx(ai, abs=TOL)
    assert hits[0]["auroc_foreign_instruction"] == pytest.approx(af, abs=TOL)


@pytest.mark.parametrize("stat,fs,im,imean,ib,ira,fm,fmean,f0", DELAYS)
def test_detection_time(res, stat, fs, im, imean, ib, ira, fm, fmean, f0):
    r = _row(res, stat, fs)
    d, w = r["detection_delay_iva_frames"], r["detection_frame_foreign_instruction"]
    assert d["median"] == im and w["median"] == fm
    assert d["mean"] == pytest.approx(imean, abs=0.05)
    assert w["mean"] == pytest.approx(fmean, abs=0.05)
    assert d["frac_before_onset"] == pytest.approx(ib, abs=TOL)
    assert r["episode_recall_iva_injection_after_onset"] == pytest.approx(ira, abs=TOL)
    assert w["frac_at_onset"] == pytest.approx(f0, abs=TOL)
    assert w["frac_before_onset"] == 0.0
