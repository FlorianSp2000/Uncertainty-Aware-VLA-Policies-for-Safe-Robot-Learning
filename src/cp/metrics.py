"""Frame-level detection metrics for the IVA CP experiment (pure numpy).

Labels: y=1 for false-premise (infeasible) frames, y=0 for feasible frames.
Scores: per-frame ensemble disagreement s_t (higher = more anomalous).
No sklearn dependency by design; AUROC/AP are exact rank-based implementations.
"""

import numpy as np


def _validate(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    assert labels.shape == scores.shape, f"{labels.shape} vs {scores.shape}"
    assert np.all(np.isfinite(scores)), "non-finite scores"
    assert labels.any() and (~labels).any(), "need both classes for ranking metrics"
    return labels, scores


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties assigned the group-average rank."""
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(x.size)
    sx = x[sorter]
    group_start = np.r_[True, sx[1:] != sx[:-1]]
    boundaries = np.r_[np.nonzero(group_start)[0], x.size]
    avg_rank_per_group = 0.5 * (boundaries[:-1] + boundaries[1:] - 1) + 1.0
    dense = np.cumsum(group_start) - 1
    return avg_rank_per_group[dense][inv]


def auroc(labels, scores) -> float:
    """Mann-Whitney AUROC with exact tie handling."""
    labels, scores = _validate(labels, scores)
    ranks = _average_ranks(scores)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def auroc_neg_pos(neg, pos) -> float:
    """AUROC of `pos` (y=1) against `neg` (y=0)."""
    neg, pos = np.asarray(neg, dtype=np.float64).ravel(), np.asarray(pos, dtype=np.float64).ravel()
    return auroc(np.r_[np.zeros(neg.size), np.ones(pos.size)], np.r_[neg, pos])


def paired_auroc_ci(neg, pos, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of auroc_neg_pos, resampling PAIRS: neg[i] and pos[i] are the same scene
    (e.g. original vs. edited frame), so pairs, not frames, are the independent unit."""
    neg, pos = np.asarray(neg, dtype=np.float64), np.asarray(pos, dtype=np.float64)
    if neg.shape != pos.shape or neg.ndim != 1:
        raise ValueError(f"paired scores must be equal-length 1-D, got {neg.shape} vs {pos.shape}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, neg.size, (n_boot, neg.size))
    vals = _auroc_rows(neg[idx], pos[idx])
    lo, hi = np.percentile(vals, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def _auroc_rows(neg: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """auroc_neg_pos of each row pair of (B, n_neg) / (B, n_pos), vectorised; exact tie-averaged
    ranks, so identical to the per-row call."""
    x = np.concatenate([neg, pos], axis=1)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite scores")
    b, n = x.shape
    sorter = np.argsort(x, axis=1, kind="mergesort")
    sx = np.take_along_axis(x, sorter, axis=1)
    start = np.concatenate([np.ones((b, 1), bool), sx[:, 1:] != sx[:, :-1]], axis=1)
    end = np.concatenate([start[:, 1:], np.ones((b, 1), bool)], axis=1)
    j = np.broadcast_to(np.arange(n), (b, n))
    first = np.maximum.accumulate(np.where(start, j, 0), axis=1)
    last = np.minimum.accumulate(np.where(end, j, n)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty_like(x)
    np.put_along_axis(ranks, sorter, 0.5 * (first + last) + 1.0, axis=1)
    n_pos, n_neg = pos.shape[1], neg.shape[1]
    u = ranks[:, n_neg:].sum(axis=1) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def average_precision(labels, scores) -> float:
    """AP = sum over descending-score thresholds of (delta recall) * precision.

    Ties are handled by grouping equal scores into one threshold step
    (matches sklearn.average_precision_score).
    """
    labels, scores = _validate(labels, scores)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(np.float64)
    s = scores[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    # last index of each tied group = the threshold's operating point
    last_of_group = np.r_[s[1:] != s[:-1], True]
    tp_g, fp_g = tp[last_of_group], fp[last_of_group]
    precision = tp_g / (tp_g + fp_g)
    recall = tp_g / labels.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def threshold_metrics(labels, flags) -> dict:
    """TPR/FPR/balanced accuracy of boolean per-frame detections against labels.

    Always reports the class counts the rates were computed over: a TPR is not
    interpretable without knowing whether it rests on 6 positives or 280.
    """
    labels = np.asarray(labels).astype(bool).ravel()
    flags = np.asarray(flags).astype(bool).ravel()
    assert labels.shape == flags.shape
    assert labels.any() and (~labels).any(), "need both classes"
    tpr = float(flags[labels].mean())
    fpr = float(flags[~labels].mean())
    return {
        "tpr": tpr,
        "fpr": fpr,
        "balanced_acc": 0.5 * (tpr + (1.0 - fpr)),
        "n_pos": int(labels.sum()),
        "n_neg": int((~labels).sum()),
    }


def partition_by_injection(fp_records: list) -> dict:
    """Split an fp-split record list by whether the episode contains any injection.

    IVA `_fp` files ship three kinds of episode, and only the first two carry
    positives:
      - scattered : many injected frames, full-length episode
      - terminal  : exactly one injected frame, at the last frame; episode is
                    truncated there
      - clean     : ZERO injected frames — a genuinely feasible episode that
                    happens to live in the _fp file

    Pooling `clean` into the test set silently inflates the FPR denominator and
    contributes no positives. Keep it separate: it is a held-out feasible set,
    which makes it the right place to measure the false-alarm rate — provided it
    was never calibrated on.
    """
    out = {"scattered": [], "terminal": [], "clean": []}
    for r in fp_records:
        feasible = np.asarray(r["task_feasible"]).astype(bool)
        inj = np.flatnonzero(~feasible)
        if inj.size == 0:
            out["clean"].append(r)
        elif inj.size == 1 and inj[0] == feasible.size - 1:
            out["terminal"].append(r)
        else:
            out["scattered"].append(r)
    return out


def injection_report(fp_records: list) -> dict:
    """Episode/frame counts per injection type. Cheap guard against silent drift."""
    parts = partition_by_injection(fp_records)
    rep = {}
    for kind, recs in parts.items():
        frames = int(sum(len(r["s_t"]) for r in recs))
        pos = int(sum(int((~np.asarray(r["task_feasible"]).astype(bool)).sum()) for r in recs))
        rep[kind] = {"episodes": len(recs), "frames": frames, "injected_frames": pos}
    return rep


def trajectory_coverage(tp_traj_scores: list, band: dict) -> float:
    """Fraction of held-out feasible trajectories that never pierce the band.

    Should be >= 1-alpha under exchangeability (the functional-CP guarantee);
    a clearly lower value indicates a band-construction bug or distribution shift.
    """
    from say_no.cp.bands import exceeds_band

    assert len(tp_traj_scores) > 0
    covered = [not exceeds_band(s, band).any() for s in tp_traj_scores]
    return float(np.mean(covered))


def write_rows_tex(path, rows: list, fmt: str = "{:.3f}") -> None:
    """Write LaTeX body rows (label & v1 & v2 ... \\\\) per src/results convention.

    rows: list of (label:str, values:list[float|str|None]); None renders as '--'.
    """
    lines = []
    for label, values in rows:
        cells = [label]
        for v in values:
            if v is None:
                cells.append("--")
            elif isinstance(v, str):
                cells.append(v)
            else:
                cells.append(fmt.format(v))
        lines.append(" & ".join(cells) + r" \\")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
