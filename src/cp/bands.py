# Functional CP band re-implemented (not copied) after FAIL-Detect (https://github.com/CXU-TRI/FAIL-Detect) and SAFE (https://github.com/vla-safe/SAFE); split CP and CUSUM are textbook.
"""Conformal-prediction calibration for per-frame anomaly scores (pure numpy).

Two detectors over a calibration set of feasible (TP) score trajectories:

1. `split_cp_threshold` — frame-level one-sided split-CP threshold tau.
   Exchangeable frames; guarantees frame-level FPR <= alpha. The statistically
   clean headline detector.

2. `functional_cp_band` — SAFE [gu2025] / FAIL-Detect [xu2025] functional band
   with simultaneous (whole-trajectory) coverage: a feasible trajectory stays
   below `upper_t` at every t with prob >= 1-alpha. Primarily the visual for
   the SAFE-Fig-5-style figure.

Sign convention: higher score = more anomalous; only the UPPER band matters.
NOTE ON THE SIGN: the band-width statistic is D_j = max_t (s_t^j - mu_t)/varsigma(t)
(score ABOVE the mean). FAIL-Detect's paper (arXiv 2503.08558, Appendix -B) prints
(mu_t - s_t^j), a LOWER-deviation statistic, inconsistent with its own band
mu + h*varsigma and rule s_t > eta_t. Their released code computes the upper form,
i.e. ours -- follow the code, not the printed equation:
https://github.com/CXU-TRI/FAIL-Detect/blob/main/UQ_test/timeseries_cp/methods/functional_predictor.py
(get_one_sided_prediction_band, lower_bound=False)

Length alignment: the band is defined pointwise over t, so unequal-length
calibration trajectories must be aligned first. Two modes, selectable per call:

    "truncate"  T = min length; every trajectory cut to s[:T].  Discards the
                tails of long trajectories, and any test frame past T is scored
                against a flat-extended ceiling (outside the guarantee).
    "pad"       T = max length; every trajectory edge-padded (last value
                repeated) to T.  This is what SAFE does.  The profile then
                spans every observed timestep, at the cost of mu_t/varsigma(t)
                late in the episode resting on fewer distinct trajectories.

Neither is "correct" — they trade coverage horizon against profile support.
"truncate" is the default because it reproduces the original results.
"""

import numpy as np

ALIGN_MODES = ("truncate", "pad")


def _conformal_rank(n: int, alpha: float) -> int:
    """1-based order statistic index ceil((n+1)(1-alpha)); raises if unattainable."""
    assert 0.0 < alpha < 1.0, f"alpha={alpha} out of (0,1)"
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        raise ValueError(
            f"calibration set too small for alpha={alpha}: need ceil((n+1)(1-alpha))={rank} <= n={n}"
        )
    return rank


def split_cp_threshold(scores_tp_frames: np.ndarray, alpha: float = 0.15) -> float:
    """One-sided upper split-CP threshold over calibration frames.

    tau = ceil((n+1)(1-alpha))-th order statistic; P(s_new > tau) <= alpha for
    an exchangeable feasible frame.
    """
    scores = np.asarray(scores_tp_frames, dtype=np.float64).ravel()
    assert scores.size > 0, "empty calibration scores"
    assert np.all(np.isfinite(scores)), "non-finite calibration scores"
    rank = _conformal_rank(scores.size, alpha)
    return float(np.sort(scores)[rank - 1])


def _as_trajectories(tp_traj_scores: list) -> list:
    """Validate and normalise a list of score trajectories to 1-D float arrays."""
    trajs = [np.asarray(s, dtype=np.float64).ravel() for s in tp_traj_scores]
    assert len(trajs) >= 4, f"need >=4 calibration trajectories, got {len(trajs)}"
    for s in trajs:
        assert s.size > 0 and np.all(np.isfinite(s)), "empty or non-finite score trajectory"
    return trajs


def align_trajectories(trajs: list, align: str = "truncate") -> np.ndarray:
    """Stack unequal-length trajectories into an (N, T) matrix.

    "truncate" -> T = min length, tails dropped.
    "pad"      -> T = max length, each trajectory edge-padded with its last value.
    """
    if align not in ALIGN_MODES:
        raise ValueError(f"align must be one of {ALIGN_MODES}, got {align!r}")
    lengths = [s.size for s in trajs]
    if align == "truncate":
        T = min(lengths)
        return np.stack([s[:T] for s in trajs], axis=0)
    T = max(lengths)
    return np.stack([np.pad(s, (0, T - s.size), mode="edge") for s in trajs], axis=0)


MODULATIONS = ("adaptive", "std", "quantile", "constant")


def _modulation(S_A: np.ndarray, mu: np.ndarray, alpha: float, kind: str) -> np.ndarray:
    """Per-timestep scale varsigma(t) for the band / z-score.

    "adaptive"  FAIL-Detect Eq. 2, a trimmed MAX over calibration trajectories.
                NOT SCALE-STABLE IN N: a maximum is an extreme-value statistic, so
                varsigma grows as calibration data is added and the band widens
                with more evidence. Measured on IVA turn_tap, mean varsigma rises
                ~56% from N=6 to N=25, against ~26% for std. This confounds any
                comparison of calibration sources of different size (eval, N~20,
                vs train, N~450), so prefer a stable scale when that comparison
                is the point.
    "std"       per-timestep standard deviation. Converges in N.
    "quantile"  per-timestep (1-alpha) quantile of |s - mu|. Converges in N, and
                keeps the one-sided conformal flavour of the original.
    "constant"  flat 1/T, i.e. no time conditioning.
    """
    if kind not in MODULATIONS:
        raise ValueError(f"modulation must be one of {MODULATIONS}, got {kind!r}")
    T = S_A.shape[1]
    if kind == "constant":
        return np.full(T, 1.0 / T)
    if kind == "adaptive":
        return _adaptive_modulation(S_A, mu, alpha)
    resid = np.abs(S_A - mu)
    v = resid.std(axis=0) if kind == "std" else np.quantile(resid, 1.0 - alpha, axis=0)
    if np.any(v <= 0.0):
        raise ValueError(f"degenerate {kind} modulation: zero spread at some timestep")
    return v


def _adaptive_modulation(S_A: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """FAIL-Detect Eq. 2: varsigma(t) = max_{k in H} |s_t^k - mu_t| with outlier trim.

    H drops trajectories whose max-abs-residual exceeds the (1-alpha)-quantile
    gamma; the trim is skipped when (N1+1)(1-alpha) > N1 (too few trajectories).

    See _modulation for the N-stability caveat.
    """
    N1 = S_A.shape[0]
    resid = np.abs(S_A - mu)          # (N1, T)
    traj_max = resid.max(axis=1)      # (N1,)

    if (N1 + 1) * (1.0 - alpha) > N1:
        keep = np.ones(N1, dtype=bool)
    else:
        gamma = np.sort(traj_max)[_conformal_rank(N1, alpha) - 1]
        keep = traj_max <= gamma
        assert keep.any(), "adaptive modulation trimmed all calibration trajectories"

    varsigma = resid[keep].max(axis=0)  # (T,)
    if np.any(varsigma <= 0.0):
        raise ValueError(
            "degenerate modulation: all D_A trajectories share identical scores at some timestep"
        )
    return varsigma


def functional_cp_band(
    tp_traj_scores: list,
    alpha: float = 0.15,
    modulation: str = "adaptive",
    split_seed: int = 0,
    align: str = "truncate",
) -> dict:
    """Fit the one-sided functional CP band on feasible score trajectories.

    Trajectories are aligned to a common length T (see `align`) and split
    disjointly into D_A (mean + modulation) and D_B (band width), 50/50 with a
    fixed permutation seed.

    Returns dict(mu, varsigma, h, upper, T, idx_A, idx_B, alpha, align).
    """
    trajs = _as_trajectories(tp_traj_scores)
    S = align_trajectories(trajs, align)  # (N, T)
    T = S.shape[1]

    perm = np.random.default_rng(split_seed).permutation(len(trajs))
    n_A = len(trajs) // 2
    idx_A, idx_B = perm[:n_A], perm[n_A:]
    S_A, S_B = S[idx_A], S[idx_B]

    mu = S_A.mean(axis=0)  # (T,)
    varsigma = _modulation(S_A, mu, alpha, modulation)

    # Width from D_B: D_j = max_t (s_t^j - mu_t)/varsigma(t)  [corrected sign].
    D = ((S_B - mu) / varsigma).max(axis=1)  # (N2,)
    h = float(np.sort(D)[_conformal_rank(D.size, alpha) - 1])

    return {
        "mu": mu,
        "varsigma": varsigma,
        "h": h,
        "upper": mu + h * varsigma,
        "T": T,
        "idx_A": idx_A,
        "idx_B": idx_B,
        "alpha": alpha,
        "align": align,
    }


def exceeds_band(s: np.ndarray, band: dict) -> np.ndarray:
    """Per-frame band exceedance for a full-length trajectory.

    Frames past the band's common length T are compared against upper_{T-1}.
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    upper = band["upper"]
    upper_ext = np.concatenate([upper, np.full(max(0, s.size - upper.size), upper[-1])])
    return s > upper_ext[: s.size]


def _extend(a: np.ndarray, n: int) -> np.ndarray:
    """Extend a per-timestep profile past its length by repeating the last value."""
    return np.concatenate([a, np.full(max(0, n - a.size), a[-1])])[:n]


def _zscores(s: np.ndarray, mu: np.ndarray, varsigma: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64).ravel()
    return (s - _extend(mu, s.size)) / _extend(varsigma, s.size)


def zscore_cp_threshold(
    tp_traj_scores: list,
    alpha: float = 0.15,
    split_seed: int = 0,
    align: str = "truncate",
    modulation: str = "adaptive",
    split_frac: float = 0.5,
    tau_alpha: float = None,
) -> dict:
    """Time-conditional FRAME-level threshold: standardize s_t by the feasible profile.

    Fixes the granularity mismatch of the functional band for sparse one-frame
    events: mu_t/varsigma(t) are fit on D_A (the feasible time profile), each
    frame is standardized to z_t=(s_t-mu_t)/varsigma(t) ("how unusual for this
    point in the episode"), and one flat order statistic q_z over all D_B frames
    thresholds z. The ceiling upper_t = mu_t + q_z*varsigma(t) tracks the
    feasible drift instead of being inflated by it.

    GUARANTEE -- weaker than split_cp_threshold's, despite the same formula.
    The calibration units are FRAMES pooled across episodes, and frames within an
    episode are dependent, so they are not exchangeable with a frame from a fresh
    episode. This is "CDF pooling" for a two-layer hierarchical model (Dunn,
    Wasserman & Ramdas, arXiv 1809.07441, Method 1 / Thm 4): coverage -> 1-alpha
    only ASYMPTOTICALLY IN THE NUMBER OF EPISODES, so the effective sample size is
    n_B episodes (~12 on IVA eval), not n_B*T frames. It says nothing at all about
    the per-EPISODE false-alarm rate -- measure that separately (ep_fpr_k1); on
    IVA it is ~5x the per-frame rate.

    `tau_alpha` additionally sets tau_z, the (1-tau_alpha) quantile of the D_A frames' z --
    the exceedance level the counting episode statistics use. It is taken from D_A, never
    D_B, so the episode quantile over D_B stays an exact split-conformal one. D_A's z are
    in-sample (mu, varsigma were fit on them), so held-out feasible frames exceed tau_z
    more often than tau_alpha; tau_z is a feature parameter, not a rate.
    """
    trajs = _as_trajectories(tp_traj_scores)
    S = align_trajectories(trajs, align)
    T = S.shape[1]

    perm = np.random.default_rng(split_seed).permutation(len(trajs))
    n_A = max(1, int(round(split_frac * len(trajs))))
    assert n_A < len(trajs), f"split_frac={split_frac} leaves no calibration episodes"
    idx_A, idx_B = perm[:n_A], perm[n_A:]
    S_A = S[idx_A]

    mu = S_A.mean(axis=0)
    varsigma = _modulation(S_A, mu, alpha, modulation)

    z_B = np.concatenate([_zscores(trajs[i], mu, varsigma) for i in idx_B])
    q_z = float(np.sort(z_B)[_conformal_rank(z_B.size, alpha) - 1])
    tau_z = None
    if tau_alpha is not None:
        assert 0.0 < tau_alpha < 1.0, f"tau_alpha={tau_alpha} out of (0,1)"
        z_A = np.concatenate([_zscores(trajs[i], mu, varsigma) for i in idx_A])
        tau_z = float(np.quantile(z_A, 1.0 - tau_alpha))

    return {
        "mu": mu,
        "varsigma": varsigma,
        "q_z": q_z,
        "upper": mu + q_z * varsigma,
        "T": T,
        "idx_A": idx_A,
        "idx_B": idx_B,
        "alpha": alpha,
        "align": align,
        "modulation": modulation,
        "tau_z": tau_z,
    }


def exceeds_zband(s: np.ndarray, zband: dict) -> np.ndarray:
    """Per-frame flags of the time-conditional z-score detector."""
    return _zscores(s, zband["mu"], zband["varsigma"]) > zband["q_z"]


EPISODE_STATISTICS = ("max", "topk_mean", "count_above", "frac_above", "cusum")
# Statistics that never decrease as the episode grows: the first frame at which the prefix
# statistic exceeds q_M is then an online alarm time, and an alarm at some frame is the
# same event as an alarm on the whole episode. topk_mean and frac_above are not.
ONLINE_STATISTICS = ("max", "count_above", "cusum")
NEEDS_TAU = ("count_above", "frac_above", "cusum")
# frac_above's tie-break must stay below the smallest gap between two distinct fractions,
# 1/(T1*T2); 1e-9 is below it for every episode up to this length.
MAX_EPISODE_LEN = 10_000


def _episode_quantile(M: np.ndarray, alpha: float, require_attainable: bool):
    """Conformal quantile of the episode statistics, or None when the caller will supply
    its own. `None` rather than a silently-loosened level: a band whose q_M is None makes
    `episode_alarm` fail loudly instead of alarming at a level nobody asked for."""
    if not require_attainable and int(np.ceil((M.size + 1) * (1 - alpha))) > M.size:
        return None
    return float(np.sort(M)[_conformal_rank(M.size, alpha) - 1])


def _tie_break(z_max):
    """Monotone map of the running max into (0, 1), added to a count so that equal counts are
    ordered by the episode's peak instead of tied. With ties the conformal quantile is only
    conservative (realised rate below alpha); ordered, it is exact again, deterministically."""
    return 0.5 + np.arctan(z_max) / np.pi


def running_statistic(z: np.ndarray, kind: str, tau: float = None) -> np.ndarray:
    """The episode statistic over every prefix z[:t+1]; element -1 is the episode's value.

    cusum is the one-sided Page CUSUM with drift tau, S_t = max(0, S_{t-1} + z_t - tau),
    in its closed form C_t - min(0, min_{s<=t} C_s), and reported as its running max.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    assert z.size > 0, "empty z trajectory"
    if kind not in ONLINE_STATISTICS:
        raise ValueError(f"{kind!r} has no online form, expected one of {ONLINE_STATISTICS}")
    if kind == "max":
        return np.maximum.accumulate(z)
    if tau is None:
        raise ValueError(f"statistic {kind!r} needs tau (set tau_alpha when calibrating)")
    if kind == "count_above":
        return np.cumsum(z > tau) + _tie_break(np.maximum.accumulate(z))
    c = np.cumsum(z - tau)
    return np.maximum.accumulate(c - np.minimum(0.0, np.minimum.accumulate(c)))


def episode_statistic(z: np.ndarray, kind: str = "max", k: int = 5, tau: float = None) -> float:
    """Collapse a per-frame z trajectory to the one number the episode alarms on.

    `max` is the natural statistic for a sparse impulse. `topk_mean` gives up some
    sensitivity to a single frame in exchange for not being set by one outlier frame.
    `count_above` counts frames with z > tau, `frac_above` divides that count by the episode
    length (so episodes of different length share one axis, which matters once the quantile
    is pooled across tasks of 92-164 frame medians), and `cusum` accumulates z - tau.
    Counts are tie-broken by the episode's max z (see _tie_break). frac_above needs the
    episode's length, so it is an end-of-episode check, not an online alarm.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    if kind == "max":
        return float(z.max())
    if kind == "topk_mean":
        return float(np.sort(z)[-min(k, z.size):].mean())
    if kind in ("count_above", "cusum"):
        return float(running_statistic(z, kind, tau)[-1])
    if kind == "frac_above":
        if tau is None:
            raise ValueError("statistic 'frac_above' needs tau (set tau_alpha when calibrating)")
        assert z.size <= MAX_EPISODE_LEN, f"episode of {z.size} frames breaks the frac tie-break"
        return float((z > tau).mean() + 1e-9 * _tie_break(z.max()))
    raise ValueError(f"unknown episode statistic {kind!r}, expected one of {EPISODE_STATISTICS}")


def episode_cp_threshold(
    tp_traj_scores: list,
    alpha: float = 0.10,
    split_seed: int = 0,
    align: str = "truncate",
    modulation: str = "adaptive",
    statistic: str = "max",
    k: int = 5,
    split_frac: float = 0.5,
    extra_calib_scores: list = None,
    require_attainable: bool = True,
    tau_alpha: float = None,
) -> dict:
    """EPISODE-level conformal threshold: at most alpha of feasible EPISODES alarm.

    `zscore_cp_threshold` calibrates on frames pooled across episodes. That controls the
    per-frame false-alarm rate and says nothing about the per-episode one: on IVA the
    realised frame rate tracks alpha to 0.006 and the episode rate is 0.853, because a
    150-frame feasible episode gets 150 independent chances to fire. Making the episode
    rate the thing that is promised requires making the EPISODE the calibration unit.

    So the statistic M_j = stat_t z_t^j is computed once per held-out feasible episode and
    the threshold is its ceil((n+1)(1-alpha)) order statistic. Episodes are exchangeable
    where frames within one are not, so the guarantee is exact, finite-sample, and about
    the quantity a deployment is judged on.

    The price is quantisation: attainable levels are multiples of 1/(n+1) in the number of
    calibration EPISODES, which is why `extra_calib_scores` exists -- z-scoring has already
    divided out the per-task level, so feasible episodes of other tasks can enlarge the
    calibration set if their M distribution is exchangeable with this task's. That is an
    assumption about the data, not a theorem: check it before using it.

    `tau_alpha` is required by the counting statistics (NEEDS_TAU); see zscore_cp_threshold.
    """
    _check_tau(statistic, tau_alpha)
    z = zscore_cp_threshold(tp_traj_scores, alpha=alpha, split_seed=split_seed,
                            align=align, modulation=modulation, split_frac=split_frac,
                            tau_alpha=tau_alpha)
    trajs = _as_trajectories(tp_traj_scores)
    M = [episode_statistic(_zscores(trajs[i], z["mu"], z["varsigma"]), statistic, k, z["tau_z"])
         for i in z["idx_B"]]
    if extra_calib_scores:
        M += [episode_statistic(_zscores(s, z["mu"], z["varsigma"]), statistic, k, z["tau_z"])
              for s in _as_trajectories(extra_calib_scores)]
    M = np.asarray(M, dtype=np.float64)
    return {**z, "q_M": _episode_quantile(M, alpha, require_attainable), "statistic": statistic,
            "k": k, "n_calib_episodes": int(M.size), "calib_M": M}


def _check_tau(statistic, tau_alpha):
    if statistic not in EPISODE_STATISTICS:
        raise ValueError(f"unknown episode statistic {statistic!r}, expected one of {EPISODE_STATISTICS}")
    if statistic in NEEDS_TAU and tau_alpha is None:
        raise ValueError(f"statistic {statistic!r} needs tau_alpha")


def _channels(eband: dict) -> list:
    """Per-channel z profiles of a single- or multi-channel episode band."""
    return eband["channels"] if "channels" in eband else [eband]


def episode_score(channel_scores: list, eband: dict) -> float:
    """The episode statistic the band alarms on, before thresholding: one frame-score
    trajectory per channel, max over channels. Threshold-free, so it is what an episode
    AUROC ranks."""
    chans = _channels(eband)
    assert len(channel_scores) == len(chans), "channel count mismatch"
    return max(episode_statistic(_zscores(s, z["mu"], z["varsigma"]), eband["statistic"],
                                 eband["k"], z["tau_z"])
               for s, z in zip(channel_scores, chans))


def first_alarm_frame(channel_scores: list, eband: dict):
    """Index of the first frame at which the prefix statistic exceeds q_M, or None.

    Defined only for ONLINE_STATISTICS, where it is None exactly when the episode does not
    alarm."""
    assert eband["q_M"] is not None, "band has no threshold; supply one before alarming"
    chans = _channels(eband)
    assert len(channel_scores) == len(chans), "channel count mismatch"
    run = np.max(np.stack([running_statistic(_zscores(s, z["mu"], z["varsigma"]),
                                             eband["statistic"], z["tau_z"])
                           for s, z in zip(channel_scores, chans)]), axis=0)
    over = np.flatnonzero(run > eband["q_M"])
    return int(over[0]) if over.size else None


def episode_alarm(s: np.ndarray, eband: dict) -> bool:
    """Whole-episode alarm of the episode-level detector."""
    assert eband["q_M"] is not None, "band has no threshold; supply one before alarming"
    return bool(episode_score([s], eband) > eband["q_M"])


def episode_cp_threshold_multi(
    channel_trajs: list,
    alpha: float = 0.10,
    split_seed: int = 0,
    align: str = "truncate",
    modulation: str = "adaptive",
    statistic: str = "max",
    k: int = 5,
    split_frac: float = 0.5,
    require_attainable: bool = True,
    tau_alpha: float = None,
) -> dict:
    """Episode-level threshold over SEVERAL score channels at once.

    Two frame scores read different failure modes -- ensemble disagreement responds to an
    instruction the model has never seen, mean Q to one it has learned is wrong for this
    scene -- and a deployment meets both. Running two detectors would mean spending alpha
    twice; calibrating on the combined statistic spends it once.

    Each channel is z-scored against its own feasible time profile, which puts channels of
    different scale and sign on one axis, and the episode statistic is the max over channels.
    The conformal quantile is then taken over that single number per calibration episode, so
    the guarantee is the ordinary split-conformal one and needs no multiplicity correction.

    `channel_trajs` is one list of feasible calibration trajectories per channel, all in the
    same episode order.
    """
    assert len(channel_trajs) >= 1, "no channels"
    _check_tau(statistic, tau_alpha)
    n = len(channel_trajs[0])
    assert all(len(c) == n for c in channel_trajs), "channels have different episode counts"

    per_channel = [
        zscore_cp_threshold(c, alpha=alpha, split_seed=split_seed, align=align,
                            modulation=modulation, split_frac=split_frac, tau_alpha=tau_alpha)
        for c in channel_trajs
    ]
    idx_B = per_channel[0]["idx_B"]
    M = np.array([
        max(episode_statistic(_zscores(channel_trajs[ci][i], z["mu"], z["varsigma"]), statistic,
                              k, z["tau_z"])
            for ci, z in enumerate(per_channel))
        for i in idx_B
    ])
    return {"channels": per_channel, "q_M": _episode_quantile(M, alpha, require_attainable),
            "statistic": statistic, "k": k, "alpha": alpha,
            "n_calib_episodes": int(M.size), "calib_M": M}


def episode_alarm_multi(channel_scores: list, eband: dict) -> bool:
    """Whole-episode alarm of the multi-channel detector; one score trajectory per channel."""
    assert eband["q_M"] is not None, "band has no threshold; supply one before alarming"
    return bool(episode_score(channel_scores, eband) > eband["q_M"])
