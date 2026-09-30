"""The episode-level conformal threshold must actually control the episode false-alarm rate.

That is the whole reason it exists: the frame-level detector's realised frame rate tracks
alpha to 0.006 and its episode rate is 0.853, and swapping the calibration unit is what is
supposed to fix it. A guarantee is worth nothing unverified, so it is checked here on
synthetic exchangeable episodes where the truth is known, not only on IVA where it cannot
be separated from the data.
"""

import numpy as np
import pytest

from say_no.cp.bands import (
    episode_alarm,
    episode_cp_threshold,
    episode_statistic,
    zscore_cp_threshold,
)


def make_episodes(n, rng, length=(80, 160), drift=0.05):
    """Exchangeable feasible-looking score trajectories of unequal length."""
    return [np.abs(rng.normal(0, 1, int(rng.integers(*length))).cumsum() * drift) + 1.0
            for _ in range(n)]


def test_episode_false_alarm_rate_tracks_alpha():
    """The realised rate on FRESH episodes must sit near alpha, not far above it.

    Averaged over many independent calibration/test draws: a single draw of 40 test
    episodes has a standard error of ~0.05 at alpha=0.2, which would swamp the effect.
    """
    alpha = 0.2
    realised = []
    for trial in range(60):
        rng = np.random.default_rng(trial)
        calib = make_episodes(40, rng)
        test = make_episodes(40, rng)
        band = episode_cp_threshold(calib, alpha=alpha, split_seed=trial)
        realised.append(np.mean([episode_alarm(s, band) for s in test]))
    assert abs(float(np.mean(realised)) - alpha) < 0.05, float(np.mean(realised))


def test_the_frame_level_detector_is_the_thing_this_fixes():
    """Same score, same alpha, same episodes: the frame detector alarms on far more
    episodes than alpha, which is the failure the episode detector exists to remove."""
    from say_no.cp.bands import exceeds_zband

    alpha = 0.2
    rng = np.random.default_rng(0)
    calib, test = make_episodes(40, rng), make_episodes(40, rng)
    zband = zscore_cp_threshold(calib, alpha=alpha, split_seed=0)
    eband = episode_cp_threshold(calib, alpha=alpha, split_seed=0)
    frame_far = np.mean([exceeds_zband(s, zband).any() for s in test])
    episode_far = np.mean([episode_alarm(s, eband) for s in test])
    assert frame_far > 2 * alpha, frame_far
    assert episode_far < frame_far


def test_an_unattainable_alpha_is_refused_not_silently_rounded():
    """With n calibration episodes no level finer than 1/(n+1) exists. Returning the
    smallest attainable one instead would report a guarantee that was never made."""
    rng = np.random.default_rng(0)
    calib = make_episodes(20, rng)  # split 50/50 -> 10 quantile episodes -> alpha >= 1/11
    episode_cp_threshold(calib, alpha=0.10)
    with pytest.raises(ValueError, match="calibration set too small"):
        episode_cp_threshold(calib, alpha=0.05)


def test_a_wider_calibration_half_buys_a_finer_alpha():
    """split_frac is the knob that trades profile quality for attainable level."""
    rng = np.random.default_rng(0)
    calib = make_episodes(20, rng)
    with pytest.raises(ValueError):
        episode_cp_threshold(calib, alpha=0.06, split_frac=0.5)
    band = episode_cp_threshold(calib, alpha=0.06, split_frac=0.2)
    assert band["n_calib_episodes"] == 16


def test_episode_statistics_are_what_they_claim():
    z = np.array([0.0, 5.0, 1.0, 2.0, 3.0])
    assert episode_statistic(z, "max") == 5.0
    assert episode_statistic(z, "topk_mean", k=2) == 4.0
    assert episode_statistic(z, "topk_mean", k=99) == z.mean()
    with pytest.raises(ValueError, match="unknown episode statistic"):
        episode_statistic(z, "median")


def test_default_split_frac_leaves_the_frame_detector_bit_identical():
    """split_frac was added to an existing function; its default must change nothing."""
    rng = np.random.default_rng(3)
    calib = make_episodes(25, rng)
    a = zscore_cp_threshold(calib, alpha=0.15, split_seed=1)
    b = zscore_cp_threshold(calib, alpha=0.15, split_seed=1, split_frac=0.5)
    assert a["q_z"] == b["q_z"]
    np.testing.assert_array_equal(a["mu"], b["mu"])
    np.testing.assert_array_equal(a["idx_A"], b["idx_A"])


def test_multichannel_spends_alpha_once_not_twice():
    """Two independent channels, each thresholded at alpha, would alarm on ~2*alpha of clean
    episodes. Calibrating the combined statistic must still come out at alpha -- that is the
    whole reason to combine rather than to run two detectors."""
    from say_no.cp.bands import episode_alarm_multi, episode_cp_threshold_multi

    alpha = 0.2
    combined, separate = [], []
    for trial in range(60):
        rng = np.random.default_rng(1000 + trial)
        calib = [make_episodes(40, rng), make_episodes(40, rng)]
        test = [make_episodes(40, rng), make_episodes(40, rng)]
        multi = episode_cp_threshold_multi(calib, alpha=alpha, split_seed=trial)
        singles = [episode_cp_threshold(c, alpha=alpha, split_seed=trial) for c in calib]
        combined.append(np.mean([episode_alarm_multi([test[0][i], test[1][i]], multi)
                                 for i in range(40)]))
        separate.append(np.mean([episode_alarm(test[0][i], singles[0])
                                 or episode_alarm(test[1][i], singles[1]) for i in range(40)]))
    assert abs(float(np.mean(combined)) - alpha) < 0.05, float(np.mean(combined))
    assert float(np.mean(separate)) > float(np.mean(combined))


def test_multichannel_rejects_ragged_input():
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError, match="different episode counts"):
        from say_no.cp.bands import episode_cp_threshold_multi
        episode_cp_threshold_multi([make_episodes(20, rng), make_episodes(19, rng)], alpha=0.2)


# ---- counting statistics: exceedance count / fraction above tau, CUSUM ------------------

from say_no.cp.bands import (EPISODE_STATISTICS, ONLINE_STATISTICS, _zscores, episode_alarm_multi,
                             episode_cp_threshold_multi, episode_score, first_alarm_frame,
                             running_statistic)

TAU_ALPHA = 0.1


@pytest.mark.parametrize("statistic", EPISODE_STATISTICS)
def test_every_episode_statistic_keeps_the_guarantee(statistic):
    """tau comes from D_A and the quantile from D_B, so each statistic is an ordinary
    split-conformal score. Counts are integers; the max tie-break is what keeps the realised
    rate AT alpha instead of below it."""
    alpha = 0.2
    realised = []
    for trial in range(60):
        rng = np.random.default_rng(2000 + trial)
        calib, test = make_episodes(40, rng), make_episodes(40, rng)
        band = episode_cp_threshold(calib, alpha=alpha, split_seed=trial, statistic=statistic,
                                    tau_alpha=TAU_ALPHA)
        realised.append(np.mean([episode_alarm(s, band) for s in test]))
    assert abs(float(np.mean(realised)) - alpha) < 0.05, (statistic, float(np.mean(realised)))


@pytest.mark.parametrize("statistic", ["count_above", "frac_above", "cusum"])
def test_multichannel_counting_statistics_keep_the_guarantee(statistic):
    alpha = 0.2
    realised = []
    for trial in range(60):
        rng = np.random.default_rng(3000 + trial)
        calib = [make_episodes(40, rng), make_episodes(40, rng)]
        test = [make_episodes(40, rng), make_episodes(40, rng)]
        band = episode_cp_threshold_multi(calib, alpha=alpha, split_seed=trial,
                                          statistic=statistic, tau_alpha=TAU_ALPHA)
        realised.append(np.mean([episode_alarm_multi([test[0][i], test[1][i]], band)
                                 for i in range(40)]))
    assert abs(float(np.mean(realised)) - alpha) < 0.05, (statistic, float(np.mean(realised)))


def test_tau_is_fit_on_the_profile_half_only():
    """If tau depended on D_B, the D_B statistics would no longer be exchangeable with a
    fresh episode's. Replacing every D_B trajectory must leave tau untouched."""
    rng = np.random.default_rng(5)
    calib = make_episodes(30, rng)
    z = zscore_cp_threshold(calib, alpha=0.2, split_seed=0, tau_alpha=TAU_ALPHA)
    swapped = list(calib)
    for i in z["idx_B"]:
        swapped[i] = calib[i] * 3.0 + 7.0
    z2 = zscore_cp_threshold(swapped, alpha=0.2, split_seed=0, tau_alpha=TAU_ALPHA)
    assert z["tau_z"] == z2["tau_z"]
    assert z["q_z"] != z2["q_z"]


def test_counting_statistics_are_what_they_claim():
    z = np.array([0.0, 2.0, -1.0, 3.0, 0.5])
    tau = 1.0
    count = episode_statistic(z, "count_above", tau=tau)
    assert int(count) == 2 and 0.0 < count - 2 < 1.0
    frac = episode_statistic(z, "frac_above", tau=tau)
    assert frac == pytest.approx(0.4, abs=1e-8)
    # CUSUM with drift tau against the textbook recursion
    s, best = 0.0, 0.0
    for v in z:
        s = max(0.0, s + v - tau)
        best = max(best, s)
    assert episode_statistic(z, "cusum", tau=tau) == pytest.approx(best)
    with pytest.raises(ValueError, match="needs tau"):
        episode_statistic(z, "count_above")
    with pytest.raises(ValueError, match="needs tau_alpha"):
        episode_cp_threshold(make_episodes(20, np.random.default_rng(0)), alpha=0.2,
                             statistic="cusum")


def test_equal_counts_are_ordered_by_the_peak():
    tau = 1.0
    low, high = np.array([2.0, 0.0, 0.0]), np.array([9.0, 0.0, 0.0])
    assert episode_statistic(high, "count_above", tau=tau) > episode_statistic(low, "count_above", tau=tau)
    assert episode_statistic(high, "frac_above", tau=tau) > episode_statistic(low, "frac_above", tau=tau)
    # ... but never enough to overturn a larger count
    more = np.array([1.5, 1.5, 0.0])
    assert episode_statistic(more, "count_above", tau=tau) > episode_statistic(high, "count_above", tau=tau)


@pytest.mark.parametrize("statistic", ONLINE_STATISTICS)
def test_the_alarm_time_is_consistent_with_the_episode_alarm(statistic):
    """An online statistic never decreases, so 'alarms at some frame' and 'alarms on the
    episode' are the same event, and the prefix value at the end is the episode value."""
    rng = np.random.default_rng(11)
    calib, test = make_episodes(40, rng), make_episodes(40, rng)
    band = episode_cp_threshold(calib, alpha=0.3, split_seed=0, statistic=statistic,
                                tau_alpha=TAU_ALPHA)
    for s in test + [s + 5.0 for s in test[:5]]:
        z = _zscores(s, band["mu"], band["varsigma"])
        run = running_statistic(z, statistic, band["tau_z"])
        assert np.all(np.diff(run) >= 0)
        assert run[-1] == pytest.approx(episode_score([s], band))
        t = first_alarm_frame([s], band)
        assert (t is not None) == episode_alarm(s, band)
