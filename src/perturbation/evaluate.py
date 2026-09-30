"""AUROC tables, operating points and member-subset curves over ModelDumps.

Every comparison is paired: condition c of trajectory i against `orig` of trajectory i, so the
bootstrap resamples trajectories (`paired_auroc_ci`). Positive class = perturbed (infeasible).
"""
from __future__ import annotations

import numpy as np

from say_no.cp.bands import split_cp_threshold
from say_no.cp.metrics import auroc_neg_pos, paired_auroc_ci, threshold_metrics
from say_no.perturbation.load import ModelDump
from say_no.perturbation.scores import FRAMES, crossfit_folds, member_q, plain_readouts, spread, z_readouts


def detector_frame(s: np.ndarray, frame: str) -> np.ndarray:
    """(n, 2) per-frame detector scores -> first / last / both (= mean of the two); (n, 1) -> both."""
    if s.shape[1] == 1:
        if frame != "both":
            raise ValueError("single-score detector has frame 'both' only")
        return s[:, 0]
    return {"first": s[:, 0], "last": s[:, 1], "both": s.mean(axis=1)}[frame]


def auroc_cell(neg: np.ndarray, pos: np.ndarray, n_boot: int, seed: int) -> dict:
    return {"auroc": auroc_neg_pos(neg, pos), "ci95": list(paired_auroc_ci(neg, pos, n_boot=n_boot, seed=seed))}


def paired_scores(model: ModelDump, cond: str, frame: str, folds: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """readout -> (scores on originals, scores under `cond`) at `frame`."""
    if model.kind == "detector":
        if frame not in model.frames:
            raise ValueError(f"{model.label}: frame {frame!r} not available (has {model.frames})")
        return {model.readout: (detector_frame(model.score["orig"], frame), detector_frame(model.score[cond], frame))}
    a, b = plain_readouts(model.q["orig"], frame), plain_readouts(model.q[cond], frame)
    za, zb = z_readouts(model.q["orig"], model.q[cond], frame, folds)
    out = {k: (a[k], b[k]) for k in a}
    out.update({k: (za[k], zb[k]) for k in za})
    return out


def condition_frame_table(model: ModelDump, n_boot: int, seed: int, n_folds: int,
                          conditions: list[str] | None = None) -> dict:
    """{cond: {frame: {readout: {auroc, ci95}, single_member_min, single_member_max}}}."""
    folds = crossfit_folds(len(model.trajectory_ids), n_folds, seed)
    frames = model.frames
    out = {}
    for c in model.conditions:
        if c == "orig" or (conditions is not None and c not in conditions):
            continue
        out[c] = {}
        for f in frames:
            s = paired_scores(model, c, f, folds)
            row = {k: auroc_cell(*v, n_boot, seed) for k, v in s.items() if not k.startswith("neg_q_")}
            single = [auroc_neg_pos(*v) for k, v in s.items() if k.startswith("neg_q_")]
            if single:
                row["single_member_min"], row["single_member_max"] = float(min(single)), float(max(single))
            out[c][f] = row
    return out


def per_frame_shift(model: ModelDump) -> dict:
    """Paired change of the member-mean Q per frame position (cond - orig): tests whether
    perturbed inputs regress toward the population mean (first frame up, last frame down)."""
    out = {}
    for c in model.conditions:
        if c == "orig":
            continue
        row = {}
        for f in ("first", "last"):
            a, b = member_q(model.q["orig"], f).mean(1), member_q(model.q[c], f).mean(1)
            d = b - a
            row[f] = {"orig_mean": float(a.mean()), "cond_mean": float(b.mean()),
                      "mean_shift": float(d.mean()), "frac_up": float((d > 0).mean())}
        both_orig = member_q(model.q["orig"], "both").mean(1)
        row["pooled_orig_mean_q"] = float(both_orig.mean())
        row["frac_first_up_and_last_down"] = float(np.mean(
            (member_q(model.q[c], "first").mean(1) > member_q(model.q["orig"], "first").mean(1)) &
            (member_q(model.q[c], "last").mean(1) < member_q(model.q["orig"], "last").mean(1))))
        out[c] = row
    return out


def operating_point(cal: np.ndarray, neg: np.ndarray, pos: np.ndarray, alpha: float) -> dict:
    """Split-CP threshold at level alpha on calibration originals `cal`; TPR on `pos`, realised
    FPR on `neg` (test originals)."""
    tau = split_cp_threshold(cal, alpha)
    m = threshold_metrics(np.r_[np.zeros(neg.size), np.ones(pos.size)], np.r_[neg, pos] > tau)
    return {"threshold": tau, "tpr": m["tpr"], "fpr": m["fpr"], "n_cal": int(cal.size),
            "n_pos": m["n_pos"], "n_neg": m["n_neg"]}


def split_half_operating_point(neg: np.ndarray, pos: np.ndarray, alpha: float, n_splits: int, seed: int) -> dict:
    """Calibrate on the originals of a random half of the pairs, evaluate on the other half's pairs."""
    if neg.shape != pos.shape:
        raise ValueError("paired scores expected")
    rng = np.random.default_rng(seed)
    n = neg.size
    tpr, fpr = [], []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, test = perm[: n // 2], perm[n // 2:]
        r = operating_point(neg[cal], neg[test], pos[test], alpha)
        tpr.append(r["tpr"]); fpr.append(r["fpr"])
    return {"tpr_mean": float(np.mean(tpr)), "tpr_min": float(np.min(tpr)), "tpr_max": float(np.max(tpr)),
            "fpr_mean": float(np.mean(fpr)), "fpr_min": float(np.min(fpr)), "fpr_max": float(np.max(fpr)),
            "n_splits": n_splits, "n_cal": n // 2, "n_test_pairs": n - n // 2}


def member_subsets(model: ModelDump, cond: str, sizes: list[int], n_subsets: int, seed: int) -> dict:
    """AUROC of spread and -mean Q (frame 'both') over random member subsets of each size."""
    rng = np.random.default_rng(seed)
    a, b = member_q(model.q["orig"], "both"), member_q(model.q[cond], "both")
    m = a.shape[1]
    out = {}
    for k in sizes:
        if not 1 <= k <= m:
            raise ValueError(f"subset size {k} outside 1..{m}")
        subsets = [np.arange(m)] if k == m else [rng.choice(m, k, replace=False) for _ in range(n_subsets)]
        vals = {"neg_mean_q": [auroc_neg_pos(-a[:, s].mean(1), -b[:, s].mean(1)) for s in subsets]}
        if k >= 2:
            vals["spread"] = [auroc_neg_pos(spread(a[:, s]), spread(b[:, s])) for s in subsets]
        out[str(k)] = {r: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                           "n_subsets": len(v)} for r, v in vals.items()}
    return out


def shift_spearman(det: ModelDump, ens: ModelDump, cond: str, n_boot: int, seed: int) -> dict:
    """Spearman rho [CI] across trajectories between the paired shift (cond - orig) of a detector's
    score and of the ensemble spread, frame 'both': same signal -> rho near 1."""
    from say_no.baselines.embedding import spearman_ci

    if not np.array_equal(det.trajectory_ids, ens.trajectory_ids):
        raise ValueError(f"{det.label} and {ens.label} rows differ")
    d = detector_frame(det.score[cond], "both") - detector_frame(det.score["orig"], "both")
    e = spread(member_q(ens.q[cond], "both")) - spread(member_q(ens.q["orig"], "both"))
    r, lo, hi = spearman_ci(d, e, n_boot=n_boot, seed=seed)
    return {"rho": r, "ci95": [lo, hi], "n": int(d.size)}
