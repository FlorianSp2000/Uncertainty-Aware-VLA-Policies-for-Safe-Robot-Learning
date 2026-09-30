"""Leakage-free K-fold evaluation of the episode-level detector on three-way-prompt dumps.

A three-way-prompt dump (external/V-GPS/experiments/eval_iva_three_way_prompt.py) scores every
eval false-premise-split episode, frame by frame, under three instructions on identical pixels:

    true      the episode's own instruction                  -> feasible episode (negative)
    inj       IVA's injected false premise, at its frames     -> IVA-injection positive
    other_j   the instruction of another task j, every frame  -> wrong-scene positive

Calibration is K-fold over the episodes under their own instruction; an episode is never scored
by a detector its own trajectory helped calibrate. Everything here is shared by
src/scripts/cp/analyse_three_way_prompt.py (the operating-point grid) and
src/scripts/cp/analyse_episode_statistics.py (the thesis tables and figure).
"""

from collections import defaultdict

import numpy as np

from say_no.cp.bands import (ONLINE_STATISTICS, _channels, _conformal_rank, _zscores,
                             episode_cp_threshold, episode_cp_threshold_multi, episode_score,
                             first_alarm_frame, split_cp_threshold)
from say_no.cp.metrics import auroc_neg_pos

# Sample standard deviation across ensemble members (ddof = 1), as in every thesis table.
DDOF = 1

# A calibration episode has to be long enough to say something about the feasible time
# profile. The fp split ends a terminal-refusal episode AT the refusal, leaving a handful of
# 2-12 frame episodes against task medians of 92-164; under align="truncate" a single one of
# them collapses mu_t and varsigma(t) to its own length and the z-score stops conditioning on
# time at all. They are excluded from CALIBRATION only -- every episode is still scored.
MIN_CALIB_LEN = 20


def episode_scores(record, statistic="disagreement"):
    """-> (true, inj, other, injected) per-frame scores; `other` is (n_swapped, T).

    `disagreement` is `std_Q`, the project's score. `neg_mean_q` is `-mean_m Q_m`: higher
    means the critics agree the episode will not end, which is how the reward design encodes
    infeasibility. Both are computed from the same stored per-member Q.
    """
    if statistic == "disagreement":
        reduce = lambda a, axis: a.std(axis, ddof=DDOF)
    elif statistic == "neg_mean_q":
        reduce = lambda a, axis: -a.mean(axis)
    else:
        raise ValueError(f"unknown frame statistic {statistic!r}")
    true = reduce(record["q_members_true"].astype(np.float64), 0)
    inj = reduce(record["q_members_given"].astype(np.float64), 0)
    other = reduce(record["q_members_foreign"].astype(np.float64), 1)
    injected = ~np.asarray(record["task_feasible"]).astype(bool)
    assert other.shape == (len(record["foreign_tasks"]), true.size)
    return true, inj, other, injected


def contrast(s_given, s_foreign, exclude=None):
    """How anomalous the given instruction looks NEXT TO the alternatives, frame by frame.

    A frame's own nuisance -- the per-task floor found to be
    irreducible and un-normalisable from the score's own dynamics -- is shared by every
    prompt scored on that frame, so it cancels in the difference. Deployable: the reference
    pool is instructions the robot already knows, and no ground truth is used. `exclude`
    drops the given instruction from its own reference set when it happens to be one of them.
    """
    refs = s_foreign if exclude is None else np.delete(s_foreign, exclude, axis=0)
    return s_given - np.median(refs, axis=0)


def wilson(p, n, z=1.96):
    """Wilson score interval from a rate and the number of INDEPENDENT units behind it."""
    if n == 0:
        return None, None
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def channels_of(frame_statistic):
    return ["disagreement", "neg_mean_q"] if frame_statistic == "both" else [frame_statistic]


def prepare(records_by_task, frame_statistic, score="raw"):
    """Per task and episode: one (calibration, IVA-injection, [swapped...]) triple per channel,
    the first injected frame (None when the episode carries no injection), the injection mask."""
    assert score in ("raw", "contrast"), score

    def three_ways(sc):
        s_true, s_inj, s_other, _ = sc
        if score == "raw":
            return s_true, s_inj, list(s_other)
        return (contrast(s_true, s_other), contrast(s_inj, s_other),
                [contrast(s_other[j], s_other, exclude=j) for j in range(s_other.shape[0])])

    out = {}
    for t in sorted(records_by_task):
        recs = records_by_task[t]
        injected = [~np.asarray(r["task_feasible"]).astype(bool) for r in recs]
        out[t] = {
            "scores": [[three_ways(episode_scores(r, ch)) for ch in channels_of(frame_statistic)]
                       for r in recs],
            "injected": injected,
            "onset": [int(np.flatnonzero(m)[0]) if m.any() else None for m in injected],
            "episode_id": [int(r["episode_id"]) for r in recs],
        }
    return out


def fold_bands(per_task, *, seed, fold, n_folds, alpha, statistic, k, tau_alpha, align,
               min_calib_len, pool_tasks):
    """Fit every task's band for one (seed, fold). -> ({task: band}, {task: held-out idx},
    {task: calibration idx}), or None when alpha is unattainable on this calibration set.

    `pool_tasks` keeps each task's own time profile but takes the conformal quantile over the
    episode statistics of ALL tasks' calibration halves: ~90 episodes instead of ~10, which is
    what makes levels below 1/11 attainable. The per-task construction is exact; the pooled
    one is only approximately MARGINALLY valid -- it assumes the z-scored statistic is
    exchangeable across tasks, which the per-task medians contradict (per-task medians) -- and
    is not conditionally valid per task.
    """
    n_channels = len(per_task[next(iter(per_task))]["scores"][0])
    bands, holds, keeps = {}, {}, {}
    n_calib_total = 0
    for task, d in per_task.items():
        n = len(d["scores"])
        folds = np.array_split(np.random.default_rng(seed).permutation(n), n_folds)
        hold = sorted(folds[fold].tolist())
        keep = [i for i in range(n)
                if i not in set(hold) and len(d["scores"][i][0][0]) >= min_calib_len]
        assert keep, "empty calibration fold"
        # Same expression bands.zscore_cp_threshold uses for its A/B split; an approximation
        # of it here would let an unattainable alpha through.
        n_b = len(keep) - max(1, int(round(0.5 * len(keep))))
        if not pool_tasks and int(np.ceil((n_b + 1) * (1 - alpha))) > n_b:
            return None
        n_calib_total += n_b
        calib = [[d["scores"][i][ci][0] for i in keep] for ci in range(n_channels)]
        # In pooled mode the per-task quantile is discarded, so an unattainable level here is
        # not an error -- the pooled calibration set is what has to support it.
        kw = dict(alpha=alpha, split_seed=seed, align=align, statistic=statistic, k=k,
                  require_attainable=not pool_tasks, tau_alpha=tau_alpha)
        bands[task] = (episode_cp_threshold(calib[0], **kw) if n_channels == 1
                       else episode_cp_threshold_multi(calib, **kw))
        holds[task], keeps[task] = hold, keep
    if pool_tasks:
        if int(np.ceil((n_calib_total + 1) * (1 - alpha))) > n_calib_total:
            return None
        pooled = np.sort(np.concatenate([b["calib_M"] for b in bands.values()]))
        q = float(pooled[_conformal_rank(pooled.size, alpha) - 1])
        bands = {t: {**b, "q_M": q} for t, b in bands.items()}
    return bands, holds, keeps


def _delay_summary(delays, n_scored):
    """Alarm frame minus onset frame over the alarmed positives; negative = before onset."""
    d = np.asarray(delays, dtype=np.float64)
    if d.size == 0:
        return None
    return {"n_alarmed": int(d.size), "n_scored": int(n_scored),
            "median": float(np.median(d)), "mean": float(d.mean()),
            "frac_before_onset": float((d < 0).mean()), "frac_at_onset": float((d == 0).mean())}


def operating_point(records_by_task, *, alpha, statistic, k, seeds, n_folds=5, score="raw",
                    frame_statistic="disagreement", pool_tasks=False, align="truncate",
                    min_calib_len=MIN_CALIB_LEN, tau_alpha=None):
    """Episode-level detector, measured on both wrong-instruction regimes.

    Rates are over held-out scorings, pooled over (seed, fold); seeds re-score the SAME
    episodes, so every interval is on the number of DISTINCT episodes. The wrong-scene recall
    pools 8 swapped prompts per episode whose alarms are near-perfectly correlated
    (ICC ~ 1), so its interval is on the 224 distinct episodes too -- the
    conservative bound on the design effect -- not on the 1792 scorings.

    Episode AUROC is threshold-free: the episode statistic of held-out negatives against
    held-out positives, pooled over tasks and folds, one AUROC per seed.
    """
    per_task = prepare(records_by_task, frame_statistic, score)
    tasks = list(per_task)
    online = statistic in ONLINE_STATISTICS
    far, rec_iva, rec_iva_after, rec_foreign = [], [], [], []
    delays_iva, delays_foreign = [], []
    by_task = defaultdict(lambda: ([], [], []))
    auroc_iva, auroc_foreign, n_folds_fitted = [], [], 0
    # One deployment calibrates once; the pooled FAR hides how far a single calibration lands.
    far_by_calibration = []
    for seed in seeds:
        neg_s, iva_s, for_s = [], [], []
        for f in range(n_folds):
            fitted = fold_bands(per_task, seed=seed, fold=f, n_folds=n_folds, alpha=alpha,
                                statistic=statistic, k=k, tau_alpha=tau_alpha, align=align,
                                min_calib_len=min_calib_len, pool_tasks=pool_tasks)
            if fitted is None:
                return None
            bands, holds, _ = fitted
            n_folds_fitted += 1
            far_start = len(far)
            for task in tasks:
                band, d = bands[task], per_task[task]
                for i in holds[task]:
                    sc = d["scores"][i]
                    ch = lambda cond, j=None: [c[cond] if j is None else c[2][j] for c in sc]
                    m = episode_score(ch(0), band)
                    neg_s.append(m)
                    far.append(m > band["q_M"])
                    by_task[task][0].append(m > band["q_M"])
                    if d["onset"][i] is not None:
                        m = episode_score(ch(1), band)
                        iva_s.append(m)
                        rec_iva.append(m > band["q_M"])
                        by_task[task][1].append(m > band["q_M"])
                        if online:
                            t = first_alarm_frame(ch(1), band)
                            assert (t is not None) == (m > band["q_M"])
                            rec_iva_after.append(t is not None and t >= d["onset"][i])
                            if t is not None:
                                delays_iva.append(t - d["onset"][i])
                    for j in range(len(sc[0][2])):
                        m = episode_score(ch(2, j), band)
                        for_s.append(m)
                        rec_foreign.append(m > band["q_M"])
                        by_task[task][2].append(m > band["q_M"])
                        if online:
                            t = first_alarm_frame(ch(2, j), band)
                            if t is not None:
                                delays_foreign.append(t)
            far_by_calibration.append(float(np.mean(far[far_start:])))
        auroc_iva.append(auroc_neg_pos(neg_s, iva_s))
        auroc_foreign.append(auroc_neg_pos(neg_s, for_s))
    n_seeds = len(seeds)
    n_feasible = len(far) // n_seeds
    n_iva = len(rec_iva) // n_seeds
    return {
        "alpha": alpha,
        "statistic": f"topk_mean{k}" if statistic == "topk_mean" else statistic,
        "tau_alpha": tau_alpha,
        "score": score, "frame_statistic": frame_statistic, "pool_tasks": pool_tasks,
        "episode_far": float(np.mean(far)),
        "episode_far_by_calibration": far_by_calibration,
        "episode_recall_iva_injection": float(np.mean(rec_iva)),
        "episode_recall_foreign_instruction": float(np.mean(rec_foreign)),
        "far_ci": wilson(np.mean(far), n_feasible),
        "recall_iva_ci": wilson(np.mean(rec_iva), n_iva),
        "recall_foreign_ci": wilson(np.mean(rec_foreign), n_feasible),
        "episode_auroc_iva_injection": float(np.mean(auroc_iva)),
        "episode_auroc_foreign_instruction": float(np.mean(auroc_foreign)),
        "episode_auroc_iva_injection_by_seed": auroc_iva,
        "episode_auroc_foreign_instruction_by_seed": auroc_foreign,
        # Recall counting only alarms at or after the first injected frame: an alarm before
        # onset fired on frames identical to the feasible episode's, so it is a false alarm
        # that happens to land in a positive episode.
        "episode_recall_iva_injection_after_onset": (float(np.mean(rec_iva_after))
                                                     if online else None),
        "detection_delay_iva_frames": _delay_summary(delays_iva, len(rec_iva)) if online else None,
        "detection_frame_foreign_instruction": (_delay_summary(delays_foreign, len(rec_foreign))
                                                if online else None),
        "n_feasible_episodes": n_feasible,
        "n_iva_episodes": n_iva,
        "n_foreign_episode_scorings": len(rec_foreign) // n_seeds,
        "n_eff_foreign": n_feasible,
        "n_seeds": n_seeds, "n_calibrations": n_folds_fitted,
        "align": align, "min_calib_len": min_calib_len,
        "n_episodes_excluded_from_calibration": sum(
            1 for d in per_task.values() for s in d["scores"] if len(s[0][0]) < min_calib_len),
        # A 9-task mean never stands alone in this repo -- push_buttons is a known dataset
        # artefact and a failing task must not be able to hide inside the aggregate.
        "per_task": {t: {"far": float(np.mean(v[0])),
                         "recall_iva": float(np.mean(v[1])) if v[1] else None,
                         "recall_foreign": float(np.mean(v[2])),
                         "n_feasible_episodes": len(v[0]) // n_seeds,
                         "n_iva_episodes": len(v[1]) // n_seeds}
                     for t, v in sorted(by_task.items())},
    }


def _frame_z(sc_channels, band, cond, j=None):
    """Max-over-channels z trajectory of one scoring under a fitted band."""
    trajs = [c[cond] if j is None else c[2][j] for c in sc_channels]
    return np.max(np.stack([_zscores(s, z["mu"], z["varsigma"])
                            for s, z in zip(trajs, _channels(band))]), axis=0)


def frame_level(records_by_task, *, alpha, seeds, n_folds=5, frame_statistic="both",
                align="truncate", min_calib_len=MIN_CALIB_LEN, pool_tasks=True):
    """The frame-level view of the same detector, on the same folds.

    Frame score = max over channels of the held-out z_t. Frame threshold = split-conformal
    (1-alpha) quantile of the calibration half-B frames' score, pooled across tasks -- the
    construction the deployed frame-level layer uses, whose guarantee is per FRAME and only
    asymptotic (frames within an episode are dependent; bands.zscore_cp_threshold).

      frame negatives   feasible frames (task_feasible) of every held-out episode, true prompt
      frame positives   IVA: the injected frames, injection prompt; wrong scene: every frame
                        of every held-out episode under each swapped prompt
      episode columns   the same frame threshold, an episode alarming when ANY frame exceeds it
    """
    per_task = prepare(records_by_task, frame_statistic)
    # exceedance counts: [n_over, n] per frame class, pooled over folds and seeds
    cnt = {"neg": [0, 0], "iva": [0, 0], "foreign": [0, 0]}
    au_iva, au_for, ep_far, ep_iva, ep_for = [], [], [], [], []
    for seed in seeds:
        neg_all, iva_all, for_all = [], [], []
        for f in range(n_folds):
            bands, holds, keeps = fold_bands(
                per_task, seed=seed, fold=f, n_folds=n_folds, alpha=alpha, statistic="max",
                k=0, tau_alpha=None, align=align, min_calib_len=min_calib_len, pool_tasks=True)
            calib_B = {t: [keeps[t][j] for j in _channels(bands[t])[0]["idx_B"]] for t in bands}
            z_B = {t: np.concatenate([_frame_z(per_task[t]["scores"][i], bands[t], 0)
                                      for i in calib_B[t]]) for t in bands}
            if pool_tasks:
                q_all = split_cp_threshold(np.concatenate(list(z_B.values())), alpha)
                q_frame = {t: q_all for t in bands}
            else:
                q_frame = {t: split_cp_threshold(z_B[t], alpha) for t in bands}
            for t, band in bands.items():
                d, q = per_task[t], q_frame[t]
                for i in holds[t]:
                    sc, inj = d["scores"][i], d["injected"][i]
                    z0 = _frame_z(sc, band, 0)
                    neg_all.append(z0[~inj])
                    cnt["neg"][0] += int((z0[~inj] > q).sum()); cnt["neg"][1] += int((~inj).sum())
                    ep_far.append(bool((z0 > q).any()))
                    if inj.any():
                        z1 = _frame_z(sc, band, 1)
                        iva_all.append(z1[inj])
                        cnt["iva"][0] += int((z1[inj] > q).sum()); cnt["iva"][1] += int(inj.sum())
                        ep_iva.append(bool((z1 > q).any()))
                    for j in range(len(sc[0][2])):
                        z2 = _frame_z(sc, band, 2, j)
                        for_all.append(z2)
                        cnt["foreign"][0] += int((z2 > q).sum()); cnt["foreign"][1] += z2.size
                        ep_for.append(bool((z2 > q).any()))
        neg = np.concatenate(neg_all)
        au_iva.append(auroc_neg_pos(neg, np.concatenate(iva_all)))
        au_for.append(auroc_neg_pos(neg, np.concatenate(for_all)))
    n_seeds = len(seeds)
    return {"alpha": alpha, "frame_statistic": frame_statistic, "pool_tasks": pool_tasks,
            "frame_auroc_iva_injection": float(np.mean(au_iva)),
            "frame_auroc_foreign_instruction": float(np.mean(au_for)),
            "frame_auroc_iva_injection_by_seed": au_iva,
            "frame_auroc_foreign_instruction_by_seed": au_for,
            "frame_fpr": cnt["neg"][0] / cnt["neg"][1],
            "frame_tpr_iva_injection": cnt["iva"][0] / cnt["iva"][1],
            "frame_tpr_foreign_instruction": cnt["foreign"][0] / cnt["foreign"][1],
            "n_feasible_frames": cnt["neg"][1] // n_seeds,
            "n_injected_frames": cnt["iva"][1] // n_seeds,
            "n_foreign_frames": cnt["foreign"][1] // n_seeds,
            "episode_far_any_frame": float(np.mean(ep_far)),
            "episode_recall_iva_any_frame": float(np.mean(ep_iva)),
            "episode_recall_foreign_any_frame": float(np.mean(ep_for)),
            "n_feasible_episodes": len(ep_far) // n_seeds,
            "n_iva_episodes": len(ep_iva) // n_seeds}
