"""Per-trajectory readouts of a Q-ensemble, higher = more infeasible.

A member's value at a frame is its Q averaged over the two inner heads; "both" averages the
first and last frame (the definition of the Sept-2025 inpainting evaluation and of the paper).

Time-corrected magnitude: Q depends strongly on how far the frame is from the goal (first frame
low, last frame near 0), so one threshold on raw Q mixes two populations. z_f = (Qbar_f - mu_f) /
sigma_f standardises the member mean per frame position f with mu_f, sigma_f estimated on ORIGINAL
frames of other trajectories only (K-fold cross-fitting over trajectories), never on the scored
trajectory itself. For "both" the two per-frame z are averaged (-z) or their magnitudes are
averaged (|z|). z_gap = (z_first - z_last) / 2 is the signed two-frame reading the
"perturbed inputs regress toward the average Q" hypothesis predicts (first frame up, last down).
"""
from __future__ import annotations

import numpy as np

FRAMES = ("first", "last", "both")
DDOF = 1  # sample std across members, as everywhere else in the project
ENSEMBLE_READOUTS = ("spread", "neg_mean_q", "neg_z", "abs_z", "z_gap")


def member_q(q: np.ndarray, frame: str) -> np.ndarray:
    """(n, M, H, 2) -> (n, M): each member's Q at `frame`, averaged over heads."""
    if q.ndim != 4 or q.shape[3] != 2:
        raise ValueError(f"expected (n, members, heads, 2), got {q.shape}")
    per = q.mean(axis=2)
    if frame == "first":
        return per[..., 0]
    if frame == "last":
        return per[..., 1]
    if frame == "both":
        return per.mean(axis=-1)
    raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")


def spread(m: np.ndarray) -> np.ndarray:
    if m.shape[1] < 2:
        raise ValueError("spread needs >= 2 members")
    return m.std(axis=1, ddof=DDOF)


def plain_readouts(q: np.ndarray, frame: str) -> dict[str, np.ndarray]:
    """spread, -mean Q, and -Q of each single member."""
    m = member_q(q, frame)
    out = {"spread": spread(m), "neg_mean_q": -m.mean(axis=1)}
    for k in range(m.shape[1]):
        out[f"neg_q_{k}"] = -m[:, k]
    return out


def crossfit_folds(n: int, k: int, seed: int) -> np.ndarray:
    """Fold index per trajectory, balanced, random with fixed seed."""
    if not 2 <= k <= n:
        raise ValueError(f"need 2 <= folds <= n, got {k} for n={n}")
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % k)


def frame_z(q_orig: np.ndarray, q_cond: np.ndarray, frame: str, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted z of the member-mean Q at a single frame, for originals and the condition.
    Trajectory i (both its original and its perturbed copy) is standardised with mu, sigma of the
    originals outside i's fold."""
    if frame not in ("first", "last"):
        raise ValueError("frame_z is per frame position")
    a = member_q(q_orig, frame).mean(axis=1)
    b = member_q(q_cond, frame).mean(axis=1)
    za, zb = np.empty_like(a), np.empty_like(b)
    for j in np.unique(folds):
        ref = a[folds != j]
        mu, sd = ref.mean(), ref.std(ddof=1)
        if not sd > 0:
            raise ValueError("zero sd of original Q in a fold")
        za[folds == j] = (a[folds == j] - mu) / sd
        zb[folds == j] = (b[folds == j] - mu) / sd
    return za, zb


def z_readouts(q_orig: np.ndarray, q_cond: np.ndarray, frame: str, folds: np.ndarray) -> tuple[dict, dict]:
    """-z, |z| (and z_gap for "both") at `frame` for originals and the condition."""
    if frame == "both":
        zf = [frame_z(q_orig, q_cond, f, folds) for f in ("first", "last")]
        neg = lambda i: -(zf[0][i] + zf[1][i]) / 2
        mag = lambda i: (np.abs(zf[0][i]) + np.abs(zf[1][i])) / 2
        gap = lambda i: (zf[0][i] - zf[1][i]) / 2
        return tuple({"neg_z": neg(i), "abs_z": mag(i), "z_gap": gap(i)} for i in (0, 1))
    za, zb = frame_z(q_orig, q_cond, frame, folds)
    return {"neg_z": -za, "abs_z": np.abs(za)}, {"neg_z": -zb, "abs_z": np.abs(zb)}
