"""Failure detection on public VLA rollouts: episode scores, AUROC and the conformal operating
point, computed from the per-episode dumps of ``experiments/eval_real_robot.py``.

Populations (from the ``pool`` field the converter wrote, see
``external/V-GPS/experiments/convert_safe_rollouts.py``):

* ``calib`` -- successful rollouts, used only to fit the conformal band
* ``test``  -- held-out successes (negatives) and every failure (positives)

Two frame scores, fixed before the analysis:
ensemble disagreement ``sigma_Q`` (std over members) and value score ``-Qbar`` (negative member
mean). The pre-registered episode score is the maximum over frames. The deployed detector
z-scores each frame against the calibration successes' time profile first (functional band,
``say_no.cp.bands``), which removes the part of the score that only tracks elapsed time.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from say_no.cp import bands, metrics
from say_no.cp.episode_eval import wilson

CHANNELS = {
    "ensemble_disagreement": lambda r: np.asarray(r["s_t"], dtype=np.float64),
    "value_score": lambda r: -np.asarray(r["q_mean_t"], dtype=np.float64),
    # Baselines, not ours. The policy's own mean action-token entropy per step (SAFE's CSVs),
    # and how far the executed action lies outside the bounds the critic's inputs are scaled to.
    "policy_token_entropy": lambda r: np.asarray(r["policy_entropy_t"], dtype=np.float64),
    "action_extremeness": lambda r: np.asarray(r["action_extremeness_t"], dtype=np.float64),
    # Steps so far with the gripper commanded open: a failed grasp keeps it open; separates
    # SAFE's failures as well as the critic does.
    "gripper_open_steps": lambda r: np.asarray(r["gripper_open_steps_t"], dtype=np.float64),
}
BASELINE_FIELDS = {"policy_token_entropy": "policy_entropy_t",
                   "action_extremeness": "action_extremeness_t",
                   "gripper_open_steps": "gripper_open_steps_t"}
OUR_CHANNELS = ("ensemble_disagreement", "value_score")
CHANNEL_SETS = (("ensemble_disagreement",), ("value_score",),
                ("ensemble_disagreement", "value_score"))
ENTROPY_COLUMN = "action/mean_token_entropy"


# Pools of the SAFE-protocol splits (external/V-GPS/experiments/split_vla_rollouts.py), which
# the analysis maps onto calib / test per evaluation view (select_view).
SPLIT_POOLS = {"holdout", "eval_seen", "unseen", "cv_calib", "cv_test"}


def select_view(dump: dict, calib_pool: str, test_pool: str) -> dict:
    """One evaluation of a SAFE-protocol dump: the successes of ``calib_pool`` calibrate, every
    episode of ``test_pool`` is test. SAFE's own view is eval_seen -> unseen; holdout (successes
    never trained on) calibrates the seen-task test, where eval_seen cannot do both jobs."""
    if calib_pool == test_pool:
        raise ValueError(f"calibration and test pool are both {calib_pool}")
    pools = {r["pool"] for r in dump["records"]}
    if not {calib_pool, test_pool} <= pools <= SPLIT_POOLS:
        raise ValueError(f"dump pools {pools}; view needs {calib_pool} and {test_pool}")
    records = ([dict(r, pool="calib") for r in dump["records"] if r["pool"] == calib_pool and r["success"]]
               + [dict(r, pool="test") for r in dump["records"] if r["pool"] == test_pool])
    return dict(dump, records=records, view=f"{calib_pool}->{test_pool}")


def load_dump(path: str | Path, view: tuple[str, str] | None = None) -> dict:
    with open(path, "rb") as f:
        dump = pickle.load(f)
    if view is not None:
        dump = select_view(dump, *view)
    recs = dump["records"]
    pools = {r["pool"] for r in recs}
    if not pools <= {"calib", "test"}:
        raise ValueError(f"{path}: pools {pools}, expected calib and/or test")
    for r in recs:
        if r["pool"] == "calib" and not r["success"]:
            raise ValueError(f"{path}: failure {r['key']} in the calibration pool")
        T = r["length"]
        if r["s_t"].shape != (T,) or r["q_mean_t"].shape != (T,):
            raise ValueError(f"{path}: {r['key']} score shapes {r['s_t'].shape} for T={T}")
        if not (np.isfinite(r["s_t"]).all() and np.isfinite(r["q_mean_t"]).all()):
            raise ValueError(f"{path}: non-finite scores in {r['key']}")
    return dump


def attach_policy_entropy(records: list, csv_dir: str | Path) -> None:
    """Add the policy's per-step mean token entropy from SAFE's CSV (``<source>.csv``)."""
    import csv
    for r in records:
        path = Path(csv_dir) / f"{r['source']}.csv"
        with open(path) as f:
            ent = np.array([float(row[ENTROPY_COLUMN]) for row in csv.DictReader(f)])
        if ent.shape != (r["length"],):
            raise ValueError(f"{path}: {ent.shape[0]} rows for an episode of {r['length']} frames")
        r["policy_entropy_t"] = ent


def attach_action_extremeness(records: list, actions_npz: str | Path, stats: dict) -> None:
    """Per step, the largest |normalised pose action| before clipping: 1 = on the p001/p999
    bound of the statistics the critic's inputs were scaled with, > 1 = outside. A detector that
    only looks at the action; it needs no critic at all."""
    acts = np.load(actions_npz)
    lo = np.asarray(stats["action"]["p001"], dtype=np.float64)[:6]
    hi = np.asarray(stats["action"]["p999"], dtype=np.float64)[:6]
    for r in records:
        a = acts[r["key"]].astype(np.float64)
        if a.shape != (r["length"], 7):
            raise ValueError(f"{r['key']}: actions {a.shape} for an episode of {r['length']} frames")
        r["action_extremeness_t"] = np.abs(2 * (a[:, :6] - lo) / (hi - lo + 1e-8) - 1).max(axis=1)
        # Open = at least half of the open command (Bridge convention 1 = open, both datasets).
        r["gripper_open_steps_t"] = np.cumsum(a[:, 6] > 0.5).astype(np.float64)


def populations(records: list) -> tuple[list, list, list]:
    """(calibration successes, test successes, test failures), each sorted by key."""
    key = lambda r: r["key"]  # noqa: E731
    calib = sorted([r for r in records if r["pool"] == "calib"], key=key)
    test_succ = sorted([r for r in records if r["pool"] == "test" and r["success"]], key=key)
    test_fail = sorted([r for r in records if r["pool"] == "test" and not r["success"]], key=key)
    if not (calib and test_succ and test_fail):
        raise ValueError(f"empty population: calib {len(calib)}, test successes "
                         f"{len(test_succ)}, test failures {len(test_fail)}")
    return calib, test_succ, test_fail


def truncate(records: list, horizon: str | int | None) -> tuple[list, int | None]:
    """Score every episode on its first ``horizon`` frames only.

    Needed where episode length itself carries the label: in the pi0-FAST LIBERO set every
    failure is a timeout and every success stops early, so a max over the whole episode gets
    more chances on failures and a plain timer separates the classes perfectly. Cutting all
    episodes at the shortest calibration success ("min_calib") asks the question an operator
    faces before any successful run has finished -- which runs will fail -- and gives the timer
    no information. The horizon comes from calibration episodes only, never from the test set.
    """
    if horizon in (None, "none"):
        return records, None
    per_task = None
    if horizon == "min_calib":
        h = min(r["length"] for r in records if r["pool"] == "calib")
    elif horizon == "calib_task_or_global":
        # Per task the shortest calibration success; a task without calibration successes (an
        # unseen task) gets the shortest calibration success overall. Calibration only, no test.
        per_cal = {}
        for r in records:
            if r["pool"] == "calib":
                per_cal[r["condition"]] = min(per_cal.get(r["condition"], 10**9), r["length"])
        h_global = min(per_cal.values())
        per_task = {r["condition"]: per_cal.get(r["condition"], h_global) for r in records}
        h = h_global
    elif horizon == "min_calib_task":
        # Each episode cut at the shortest calibration success of ITS task: a longer window where
        # a task's successes run long, still before any of that task's calibration successes ended.
        # Episodes of tasks without calibration successes are dropped.
        per_task = {}
        for r in records:
            if r["pool"] == "calib":
                per_task[r["condition"]] = min(per_task.get(r["condition"], 10**9), r["length"])
        records = [r for r in records if r["condition"] in per_task]
        h = min(per_task.values())
    else:
        h = int(horizon)
    out = []
    for r in records:
        if per_task is not None:
            h = per_task[r["condition"]]
        c = dict(r)
        c["s_t"], c["q_mean_t"] = r["s_t"][:h], r["q_mean_t"][:h]
        for f in BASELINE_FIELDS.values():
            if f in r:
                c[f] = r[f][:h]
        c["length"] = min(r["length"], h)
        out.append(c)
    return out, (per_task if per_task is not None else h)


def raw_episode_score(rec: dict, channel: str) -> float:
    """The pre-registered episode score: maximum of the frame score over the episode."""
    return float(CHANNELS[channel](rec).max())


def auroc_with_std(neg, pos, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC of failures (pos) over successes (neg), with a class-stratified bootstrap std and
    95 % percentile interval.

    Episodes are the independent unit; each class is resampled separately so every replicate
    keeps the observed class sizes."""
    neg, pos = np.asarray(neg, dtype=np.float64), np.asarray(pos, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.array([metrics.auroc_neg_pos(neg[rng.integers(0, neg.size, neg.size)],
                                           pos[rng.integers(0, pos.size, pos.size)])
                     for _ in range(n_boot)])
    return dict(auroc=metrics.auroc_neg_pos(neg, pos), std=float(boot.std(ddof=1)),
                ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                n_neg=int(neg.size), n_pos=int(pos.size))


def within_task_auroc(test_succ: list, test_fail: list, score, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """AUROC counting only (success, failure) pairs of the SAME task.

    Failures are unevenly spread over tasks (9 successes against 30-56 failures per task), so a
    score whose level differs by task could reach a pooled AUROC above 0.5 by recognising the
    failure-heavy tasks without detecting any failure. Restricting comparisons to within-task
    pairs removes that route. Pair-weighted: sum of per-task Mann-Whitney U over the sum of
    per-task pair counts."""
    tasks = sorted({r["condition"] for r in test_succ} & {r["condition"] for r in test_fail})
    per = {t: (np.array([score(r) for r in test_succ if r["condition"] == t]),
               np.array([score(r) for r in test_fail if r["condition"] == t])) for t in tasks}

    def pooled(groups):
        u = sum(metrics.auroc_neg_pos(n, p) * n.size * p.size for n, p in groups)
        return float(u / sum(n.size * p.size for n, p in groups))

    rng = np.random.default_rng(seed)
    boot = [pooled([(n[rng.integers(0, n.size, n.size)], p[rng.integers(0, p.size, p.size)])
                    for n, p in per.values()]) for _ in range(n_boot)]
    return dict(auroc=pooled(per.values()), std=float(np.std(boot, ddof=1)),
                ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                per_task={t: dict(auroc=metrics.auroc_neg_pos(n, p), n_neg=int(n.size),
                                  n_pos=int(p.size)) for t, (n, p) in per.items()})


def operating_point(calib: list, test_succ: list, test_fail: list, channels: tuple,
                    alpha: float, split_frac: float, n_split_seeds: int) -> dict:
    """Episode-level conformal detector calibrated on the calibration successes only.

    ``split_frac`` of the calibration episodes fit the time profile, the rest give the conformal
    quantile of the per-episode max z-score. Rates are averaged over ``n_split_seeds`` random
    calibration splits, Wilson intervals on the seed-averaged count over test episodes. Detection
    time = first frame at which the running max crosses the threshold, over episode length.
    """
    far, rec, det_time, thresholds, z_auroc, z_within = [], [], [], [], [], []
    for seed in range(n_split_seeds):
        band = bands.episode_cp_threshold_multi(
            [[CHANNELS[c](r) for r in calib] for c in channels], alpha=alpha, split_seed=seed,
            align="truncate", split_frac=split_frac)
        s_succ = np.array([bands.episode_score([CHANNELS[c](r) for c in channels], band)
                           for r in test_succ])
        s_fail = np.array([bands.episode_score([CHANNELS[c](r) for c in channels], band)
                           for r in test_fail])
        thresholds.append(band["q_M"])
        far.append(float(np.mean(s_succ > band["q_M"])))
        rec.append(float(np.mean(s_fail > band["q_M"])))
        z_auroc.append(metrics.auroc_neg_pos(s_succ, s_fail))
        z_within.append(within_task_auroc(
            test_succ, test_fail,
            lambda r: bands.episode_score([CHANNELS[c](r) for c in channels], band),
            n_boot=2)["auroc"])
        for r in test_fail:
            first = bands.first_alarm_frame([CHANNELS[c](r) for c in channels], band)
            if first is not None:
                det_time.append(first / r["length"])
    n_s, n_f = len(test_succ), len(test_fail)
    return dict(
        channels=list(channels), alpha=alpha, split_frac=split_frac,
        n_split_seeds=n_split_seeds, n_calib=len(calib),
        n_quantile_episodes=int(band["n_calib_episodes"]),
        threshold_mean=float(np.mean(thresholds)),
        false_alarm_rate=float(np.mean(far)), false_alarm_ci95=wilson(np.mean(far), n_s),
        false_alarm_seed_range=[float(min(far)), float(max(far))],
        recall=float(np.mean(rec)), recall_ci95=wilson(np.mean(rec), n_f),
        recall_seed_range=[float(min(rec)), float(max(rec))],
        detection_time_median=float(np.median(det_time)) if det_time else None,
        detection_time_iqr=([float(np.percentile(det_time, 25)),
                             float(np.percentile(det_time, 75))] if det_time else None),
        time_normalised_auroc=float(np.mean(z_auroc)),
        time_normalised_auroc_seed_range=[float(min(z_auroc)), float(max(z_auroc))],
        time_normalised_within_task_auroc=float(np.mean(z_within)),
    )


def analyse_dump(dump: dict, alpha: float, split_frac: float, n_split_seeds: int,
                 n_boot: int, horizon: str | int | None = None, discount: float = 0.98,
                 num_terminal: int = 3, test_tasks: tuple | None = None) -> dict:
    full_lengths = {r["key"]: r["length"] for r in dump["records"]}
    fit = critic_fit([r for r in dump["records"] if r["pool"] == "calib"], discount, num_terminal)
    records, h = truncate(dump["records"], horizon)
    calib, test_succ, test_fail = populations(records)
    if test_tasks:
        # SAFE's seen / unseen breakdown: calibration stays as it is, only the test set narrows.
        test_succ = [r for r in test_succ if r["condition"] in test_tasks]
        test_fail = [r for r in test_fail if r["condition"] in test_tasks]
        if not (test_succ and test_fail):
            raise ValueError(f"test_tasks {test_tasks} leave {len(test_succ)} successes, "
                             f"{len(test_fail)} failures")
    out = dict(run_dir=dump["run_dir"], step=dump["step"], horizon=h, critic_fit_calib=fit,
               test_tasks=list(test_tasks) if test_tasks else None,
               # Dumps written before the per-step gripper fix carry no such field; say so.
               gripper_binarisation=("per step (causal)" if dump.get("causal_gripper")
                                     else "whole episode" if "causal_gripper" in dump
                                     else "whole episode (dump predates the flag)"),
               # Baseline any detector must beat: the full episode's length. Perfect wherever
               # failures are timeouts; uninformative after truncation, where every scored
               # prefix of a failure has the same length.
               episode_length_auroc=metrics.auroc_neg_pos(
                   [full_lengths[r["key"]] for r in test_succ],
                   [full_lengths[r["key"]] for r in test_fail]),
               truncated_length_auroc=metrics.auroc_neg_pos(
                   [r["length"] for r in test_succ], [r["length"] for r in test_fail]),
               normalization=dump["normalization"], member_norms=dump["member_norms"],
               n_calib=len(calib), n_test_success=len(test_succ), n_test_failure=len(test_fail),
               tasks=sorted({r["condition"] for r in test_succ + test_fail}),
               channels={}, operating_points={})
    channels = [c for c in CHANNELS if c in OUR_CHANNELS or BASELINE_FIELDS[c] in records[0]]
    # The gripper count is constant across episodes at t = 0, which bands' functional band
    # refuses; it is scored under SAFE's protocol only (safe_protocol), which adds 1e-8.
    sets = list(CHANNEL_SETS) + [(c,) for c in channels
                                 if c not in OUR_CHANNELS and c != "gripper_open_steps"]
    for c in channels:
        score = lambda r, c=c: raw_episode_score(r, c)  # noqa: E731
        neg, pos = [score(r) for r in test_succ], [score(r) for r in test_fail]
        out["channels"][c] = dict(
            raw_max=auroc_with_std(neg, pos, n_boot=n_boot),
            raw_max_within_task=within_task_auroc(test_succ, test_fail, score, n_boot=n_boot),
            frame_score_range=[float(min(CHANNELS[c](r).min() for r in calib + test_succ + test_fail)),
                               float(max(CHANNELS[c](r).max() for r in calib + test_succ + test_fail))],
        )
    for chans in sets:
        out["operating_points"]["+".join(chans)] = operating_point(
            calib, test_succ, test_fail, chans, alpha, split_frac, n_split_seeds)
    out["safe_protocol"] = {}
    for c in channels:
        for variant, f in (("step", lambda r, c=c: CHANNELS[c](r)),
                           ("cumsum", lambda r, c=c: cumulative(CHANNELS[c](r)))):
            out["safe_protocol"][f"{c}/{variant}"] = safe_protocol(
                calib, test_succ, test_fail, f, SAFE_ALPHAS, n_split_seeds)
    return out


def discounted_step_count(length: int, discount: float, num_terminal: int) -> np.ndarray:
    """The Q a critic that fits a successful episode should output: reward -1 per step, the last
    ``num_terminal`` frames terminal (reward 0, no bootstrap) -- the finetune loader's labels."""
    n = length - min(num_terminal, length)
    t = np.arange(length)
    return np.where(t < n, -(1 - discount ** (n - t)) / (1 - discount), 0.0)


def critic_fit(records: list, discount: float, num_terminal: int) -> dict:
    """Validity gate: on successes the finetuned critic never trained on, does mean Q track the
    discounted step count? If not, the detector reading below is not a reading of a value
    function. Pearson r and RMSE over all frames, pooled."""
    q = np.concatenate([r["q_mean_t"] for r in records])
    y = np.concatenate([discounted_step_count(r["length"], discount, num_terminal) for r in records])
    return dict(n_episodes=len(records), pearson_r=float(np.corrcoef(q, y)[0, 1]),
                rmse=float(np.sqrt(np.mean((q - y) ** 2))), target_range=[float(y.min()), 0.0],
                q_range=[float(q.min()), float(q.max())])


def swap_control(own: dict, swapped: dict, channel: str) -> dict:
    """Positive control: the same episodes scored under their own instruction and under another
    task's. A critic that reads language must score the swapped instruction higher."""
    own_by = {r["key"]: raw_episode_score(r, channel) for r in own["records"]}
    sw_by = {r["key"]: raw_episode_score(r, channel) for r in swapped["records"]}
    keys = sorted(sw_by)
    missing = [k for k in keys if k not in own_by]
    if missing:
        raise ValueError(f"{len(missing)} swapped episodes absent from the own-instruction dump")
    a, b = np.array([own_by[k] for k in keys]), np.array([sw_by[k] for k in keys])
    lo, hi = metrics.paired_auroc_ci(a, b)
    # Same comparison on the episode MEAN of the frame score. The max of -Qbar sits at the
    # episode start, where the critic's estimate barely depends on the instruction, so the raw-max
    # comparison can say more about the first frames than about the language.
    own_mean = {r["key"]: float(CHANNELS[channel](r).mean()) for r in own["records"]}
    am = np.array([own_mean[k] for k in keys])
    bm = np.array([float(CHANNELS[channel](r).mean()) for r in
                   sorted(swapped["records"], key=lambda r: r["key"])])
    mlo, mhi = metrics.paired_auroc_ci(am, bm)
    return dict(channel=channel, n=len(keys), frac_swapped_higher=float(np.mean(b > a)),
                paired_auroc=metrics.auroc_neg_pos(a, b), paired_auroc_ci95=[lo, hi],
                median_own=float(np.median(a)), median_swapped=float(np.median(b)),
                mean_frac_swapped_higher=float(np.mean(bm > am)),
                mean_paired_auroc=metrics.auroc_neg_pos(am, bm), mean_paired_auroc_ci95=[mlo, mhi])


def per_step_profiles(records: list, channel: str) -> dict:
    """Median and quartiles of the frame score at each step, for the trajectory figure."""
    by = defaultdict(list)
    for r in records:
        by["success" if r["success"] else "failure"].append(CHANNELS[channel](r))
    return {k: dict(median=np.median(np.stack(v), 0), q25=np.percentile(np.stack(v), 25, 0),
                    q75=np.percentile(np.stack(v), 75, 0), n=len(v)) for k, v in by.items()}


# ---------------------------------------------------------------------------------------------
# SAFE's evaluation protocol (gu2025, arXiv 2506.09937, §5.4 / App. B.4-B.5; code
# github.com/vla-safe/SAFE: failure_prob/utils/metrics.py::eval_scores_roc_prc and
# conformal/functional_predictor.py), re-implemented so our numbers sit on their axes.
# ---------------------------------------------------------------------------------------------

# SAFE's operating point (gu2025): the alpha of every thesis table, figure and example of this campaign
OPERATING_ALPHA = 0.15
SAFE_ALPHAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]


def cumulative(s: np.ndarray) -> np.ndarray:
    """SAFE tries every baseline score both per step and as its running sum over time."""
    return np.cumsum(s)


def safe_roc_auc(test_succ: list, test_fail: list, score) -> float:
    """SAFE's headline ROC-AUC (`falert_early_roc_auc`): per rollout the max of the score up to
    the shortest rollout of its task (over all test rollouts of that task), pooled over tasks.
    ``score`` maps a record to its per-step score array."""
    recs = test_succ + test_fail
    t_min = {}
    for r in recs:
        t_min[r["condition"]] = min(t_min.get(r["condition"], 10**9), len(score(r)))
    x = [float(score(r)[:t_min[r["condition"]]].max()) for r in recs]
    y = [0] * len(test_succ) + [1] * len(test_fail)
    return metrics.auroc(np.array(y), np.array(x))


def _pad(trajs: list, T: int) -> np.ndarray:
    return np.stack([np.pad(s, (0, T - len(s)), mode="edge") for s in trajs])


def _safe_modulation(A: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """SAFE's Tfunc modulation (functional_predictor.py): FAIL-Detect's adaptive max residual
    over the untrimmed calibration series, plus 1e-8 -- which admits scores that are constant
    across calibration episodes at some step (e.g. a gripper count at t = 0), where
    bands._adaptive_modulation deliberately refuses."""
    resid = np.abs(A - mu)
    traj_max = resid.max(axis=1)
    n = len(A)
    keep = np.ones(n, dtype=bool)
    if (n + 1) * (1 - alpha) <= n:
        keep = traj_max <= np.sort(traj_max)[bands._conformal_rank(n, alpha) - 1]
    return resid[keep].max(axis=0) + 1e-8


def _safe_band(C: np.ndarray, alpha: float, seed: int, a_frac: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(mu_t, sigma_t, h) of SAFE's functional CP band from the padded calibration series C (N, T):
    shuffled split a_frac / 1 - a_frac, mu and sigma on the first part, h = SAFE's interpolated
    1 - alpha quantile of D_j = max_t (s_t - mu_t) / sigma_t on the second. Alarm: s_t >= mu_t + h sigma_t."""
    perm = np.random.default_rng(seed).permutation(len(C))
    n_a = int(a_frac * len(C))  # SAFE truncates (metrics.py), 72 -> 21 / 51
    A, B = C[perm[:n_a]], C[perm[n_a:]]
    mu = A.mean(axis=0)
    sigma = _safe_modulation(A, mu, alpha)
    h = float(np.quantile(((B - mu) / sigma).max(axis=1), 1 - alpha))
    return mu, sigma, h


def safe_band(calib: list, score, alpha: float, seed: int, a_frac: float = 0.3) -> dict:
    """The band `safe_functional_cp` alarms on, for drawing: mu_t, sigma_t and h of calibration
    split `seed`; a rollout alarms at the first step with s_t >= mu_t + h sigma_t."""
    S = [score(r) for r in calib]
    mu, sigma, h = _safe_band(_pad(S, max(len(s) for s in S)), alpha, seed, a_frac)
    return dict(mu=mu, sigma=sigma, h=h)


def safe_functional_cp(calib: list, test_succ: list, test_fail: list, score, alpha: float,
                       seed: int, a_frac: float = 0.3) -> dict:
    """One draw of SAFE's functional CP detector.

    Every series is edge-padded to the longest one. The calibration successes are shuffled and
    split a_frac / 1 - a_frac: mu_t on the first part, FAIL-Detect's adaptive modulation
    sigma_t on the same part, D_j = max_t (s_t - mu_t) / sigma_t on the second, h =
    np.quantile(D, 1 - alpha) (SAFE's interpolated quantile, not the finite-sample rank). A
    rollout is flagged at the first step with s_t >= mu_t + h sigma_t.

    T-det (SAFE): first flagged step / padded length, averaged over ALL failed rollouts with
    misses counted as 1. Also T-det over detected failures only (FIPER, FAIL-Detect style) and
    balanced time-weighted accuracy (FIPER Eq. 41 / Hide-and-Seek Eq. 12).
    """
    S_cal = [score(r) for r in calib]
    S_s, S_f = [score(r) for r in test_succ], [score(r) for r in test_fail]
    T = max(len(s) for s in S_cal + S_s + S_f)
    mu, sigma, h = _safe_band(_pad(S_cal, T), alpha, seed, a_frac)
    upper = mu + h * sigma

    def first(s):
        # Only the observed steps can alarm: a truncated episode's padded tail repeats its last
        # value past the scoring window and would cross a band that falls later (found on
        # LIBERO; inert where every episode has the full length).
        hit = np.flatnonzero(s >= upper[:len(s)])
        return int(hit[0]) if hit.size else None

    f_succ = [first(s) for s in S_s]
    f_fail = [first(s) for s in S_f]
    tpr = float(np.mean([f is not None for f in f_fail]))
    tnr = float(np.mean([f is None for f in f_succ]))
    tdet_all = float(np.mean([1.0 if f is None else f / T for f in f_fail]))
    tp = [f / T for f in f_fail if f is not None]
    twa = 0.5 * (sum(1 - x for x in tp) / len(f_fail) + tnr)
    return dict(tpr=tpr, tnr=tnr, bal_acc=0.5 * (tpr + tnr), t_det=tdet_all,
                t_det_tp=float(np.mean(tp)) if tp else None, twa=float(twa), h=h)


def safe_protocol(calib: list, test_succ: list, test_fail: list, score, alphas: list,
                  n_seeds: int) -> dict:
    """SAFE ROC-AUC plus the CP metrics at each alpha, averaged over calibration splits."""
    out = dict(roc_auc=safe_roc_auc(test_succ, test_fail, score), by_alpha={})
    for a in alphas:
        runs = [safe_functional_cp(calib, test_succ, test_fail, score, a, s) for s in range(n_seeds)]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in ("tpr", "tnr", "bal_acc", "t_det", "twa")}
        tps = [r["t_det_tp"] for r in runs if r["t_det_tp"] is not None]
        agg["t_det_tp"] = float(np.mean(tps)) if tps else None
        agg["bal_acc_seed_range"] = [float(min(r["bal_acc"] for r in runs)),
                                     float(max(r["bal_acc"] for r in runs))]
        out["by_alpha"][str(a)] = agg
    return out


def safe_alarm_curves(calib: list, test_succ: list, test_fail: list, score, alpha: float,
                      n_seeds: int, a_frac: float = 0.3) -> dict:
    """Timeliness curves (SAFE Fig. 10 style): the fraction of test failures, and of test
    successes, that have alarmed by each step, under SAFE's functional CP band. Averaged over
    calibration splits. An episode that ended before step t keeps its final state."""
    S_cal = [score(r) for r in calib]
    S_s, S_f = [score(r) for r in test_succ], [score(r) for r in test_fail]
    T = max(len(s) for s in S_cal + S_s + S_f)
    C = _pad(S_cal, T)
    det_f, det_s = [], []
    for seed in range(n_seeds):
        mu, sigma, h = _safe_band(C, alpha, seed, a_frac)
        upper = mu + h * sigma

        def alarmed(series):  # (N, T) bool: alarmed at or before t, own length only
            out = np.zeros((len(series), T), dtype=bool)
            for i, s in enumerate(series):
                hit = np.flatnonzero(s >= upper[:len(s)])
                if hit.size:
                    out[i, hit[0]:] = True
            return out
        det_f.append(alarmed(S_f).mean(axis=0))
        det_s.append(alarmed(S_s).mean(axis=0))
    return dict(t_frac=(np.arange(T) + 1) / T, failures_detected=np.mean(det_f, axis=0),
                successes_alarmed=np.mean(det_s, axis=0))


def safe_score_over_threshold(calib: list, series: list, score, alpha: float, seed: int,
                              a_frac: float = 0.3) -> np.ndarray:
    """Per step, (s_t - mu_t) / (h sigma_t) under one draw of SAFE's functional CP band: 0 = the
    calibration mean, 1 = the alarm threshold, so a rollout alarms at its first step at or above 1.
    Needs h > 0 (the threshold above the mean); all series share one length (no padding)."""
    S_cal, S = [score(r) for r in calib], [score(r) for r in series]
    lengths = {len(x) for x in S_cal + S}
    if len(lengths) != 1:
        raise ValueError(f"series of different lengths {sorted(lengths)}; the ratio is defined on a common grid only")
    mu, sigma, h = _safe_band(np.stack(S_cal), alpha, seed, a_frac)
    if h <= 0:
        raise ValueError(f"threshold h = {h} not above the calibration mean; the ratio does not order alarms")
    return (np.stack(S) - mu) / (h * sigma)
