"""Re-split a converted VLA-rollout cache by SAFE's seen / unseen-task protocol (gu2025, Sec. 5.1).

Input: a cache written by ``convert_safe_rollouts.py`` or ``convert_libero_rollouts.py`` that
holds every episode of the dataset (its own pools are ignored). Output: a new cache directory
whose ``index.json`` carries the new pools; the ``.npz`` files are relative symlinks into the
source, so no frame is decoded again.

Protocol, per fold:

* unseen tasks: the tasks are permuted once with ``--draw_seed``; fold k holds out the k-th
  block of ``--n_unseen`` tasks, so across folds no task is held out twice. Every episode of
  an unseen task -> ``unseen``.
* seen tasks: per (task, outcome) group a ``--seen_eval_frac`` share -> ``eval_seen`` (SAFE's
  D_eval-seen; its successes are SAFE's calibration set). The rest is SAFE's D_train, of which
  only the successes are used: a ``--holdout_frac`` share -> ``holdout`` (never trained on:
  validation TD loss and the separate-calibration variant), the remainder -> ``finetune``.
  D_train failures -> ``train_fail``, kept in the index but never read.

``--protocol foresight_cv`` instead reproduces Foresight's (zhang2026, App. 8) 3-fold
cross-validation over rollouts, all tasks seen: fold k of a per-(task, outcome) stratified
shuffle -> ``cv_test``; the other folds split train : val : calibration = 6 : 1 : 1 into
``finetune`` (successes), ``holdout`` (validation successes) and ``cv_calib`` (calibration
successes); failures outside the test fold -> ``train_fail``.

Run by ``slurm/split-vla-rollouts.sbatch``.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

POOLS = ("finetune", "holdout", "train_fail", "eval_seen", "unseen", "cv_calib", "cv_test")
CV_FOLDS = 3
CV_TRAIN_VAL_CAL = (6, 1, 1)


def unseen_tasks(tasks: list[str], draw_seed: int, fold: int, n_unseen: int) -> list[str]:
    if (fold + 1) * n_unseen > len(tasks):
        raise ValueError(f"fold {fold} x {n_unseen} unseen tasks exceeds {len(tasks)} tasks")
    perm = np.random.default_rng(draw_seed).permutation(sorted(tasks))
    return sorted(perm[fold * n_unseen:(fold + 1) * n_unseen].tolist())


def assign_pools(rows: list[dict], unseen: list[str], seen_eval_frac: float,
                 holdout_frac: float, split_seed: int) -> None:
    rng = np.random.default_rng(split_seed)
    for r in rows:
        if r["condition"] in unseen:
            r["pool"] = "unseen"
    groups = sorted({(r["condition"], r["success"]) for r in rows if r["condition"] not in unseen})
    for task, success in groups:
        eps = sorted([r for r in rows if r["condition"] == task and r["success"] == success],
                     key=lambda r: r["key"])
        order = rng.permutation(len(eps))
        n_eval = int(round(seen_eval_frac * len(eps)))
        n_train = len(eps) - n_eval
        n_hold = int(round(holdout_frac * n_train)) if success else 0
        for rank, i in enumerate(order):
            if rank < n_eval:
                eps[i]["pool"] = "eval_seen"
            elif not success:
                eps[i]["pool"] = "train_fail"
            elif rank < n_eval + n_hold:
                eps[i]["pool"] = "holdout"
            else:
                eps[i]["pool"] = "finetune"


def assign_pools_cv(rows: list[dict], fold: int, split_seed: int) -> None:
    rng = np.random.default_rng(split_seed)
    groups = sorted({(r["condition"], r["success"]) for r in rows})
    for task, success in groups:
        eps = sorted([r for r in rows if r["condition"] == task and r["success"] == success],
                     key=lambda r: r["key"])
        order = rng.permutation(len(eps))
        # Fold membership by rank modulo CV_FOLDS: per-group stratified, sizes differ by <= 1.
        rest = [eps[i] for rank, i in enumerate(order) if rank % CV_FOLDS != fold]
        for rank, i in enumerate(order):
            if rank % CV_FOLDS == fold:
                eps[i]["pool"] = "cv_test"
        bounds = np.cumsum(CV_TRAIN_VAL_CAL) / sum(CV_TRAIN_VAL_CAL) * len(rest)
        for j, r in enumerate(rest):
            part = int(np.searchsorted(bounds, j, side="right"))
            r["pool"] = ("train_fail" if not success
                         else ("finetune", "holdout", "cv_calib")[part])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True, help="cache holding every episode")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protocol", choices=("safe", "foresight_cv"), default="safe")
    ap.add_argument("--draw_seed", type=int, default=None, help="seeds the task permutation")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n_unseen", type=int, default=None)
    ap.add_argument("--seen_eval_frac", type=float, default=None)
    ap.add_argument("--holdout_frac", type=float, default=None)
    ap.add_argument("--split_seed", type=int, required=True)
    args = ap.parse_args()

    src, out = Path(args.source_dir).resolve(), Path(args.out_dir)
    rows = json.loads((src / "index.json").read_text())
    if any(r["corrupt"] for r in rows):
        raise ValueError(f"{src}: corrupt episodes present; this split assumes a clean cache")
    for r in rows:
        r["source_pool"] = r.pop("pool")
        if not (src / r["source_pool"] / f"{r['key']}.npz").exists():
            raise FileNotFoundError(f"{src}/{r['source_pool']}/{r['key']}.npz")

    safe_args = (args.draw_seed, args.n_unseen, args.seen_eval_frac, args.holdout_frac)
    if (args.protocol == "safe") != all(v is not None for v in safe_args):
        raise ValueError("--draw_seed/--n_unseen/--seen_eval_frac/--holdout_frac are required "
                         "for --protocol safe and meaningless for foresight_cv")
    if args.protocol == "safe":
        unseen = unseen_tasks(sorted({r["condition"] for r in rows}), args.draw_seed, args.fold,
                              args.n_unseen)
        assign_pools(rows, unseen, args.seen_eval_frac, args.holdout_frac, args.split_seed)
    else:
        if args.fold >= CV_FOLDS:
            raise ValueError(f"fold {args.fold} >= {CV_FOLDS}")
        unseen = []
        assign_pools_cv(rows, args.fold, args.split_seed)
    missing = [r["key"] for r in rows if r.get("pool") not in POOLS]
    if missing:
        raise RuntimeError(f"{len(missing)} episodes left without a pool, e.g. {missing[:3]}")
    if not any(r["pool"] == "finetune" for r in rows):
        raise RuntimeError("empty finetune pool")

    if out.exists():
        raise FileExistsError(f"{out} exists; splits are write-once")
    for pool in POOLS:
        (out / pool).mkdir(parents=True)
    for r in rows:
        dst = out / r["pool"] / f"{r['key']}.npz"
        os.symlink(os.path.relpath(src / r.pop("source_pool") / f"{r['key']}.npz", dst.parent), dst)

    (out / "index.json").write_text(json.dumps(rows, indent=1))
    (out / "split.json").write_text(json.dumps(dict(
        source_dir=str(src), protocol=args.protocol, draw_seed=args.draw_seed, fold=args.fold, n_unseen=args.n_unseen,
        seen_eval_frac=args.seen_eval_frac, holdout_frac=args.holdout_frac,
        split_seed=args.split_seed, unseen_tasks=unseen,
        pools={r["key"]: r["pool"] for r in rows}), indent=1))

    count = Counter((r["condition"], r["pool"], r["success"]) for r in rows)
    print(f"unseen tasks (fold {args.fold}): {unseen}")
    print(f"{'task':34s}" + "".join(f"{p + (' S/F' if p in ('eval_seen', 'unseen', 'cv_test') else ''):>16s}"
                                    for p in POOLS))
    for t in sorted({r["condition"] for r in rows}):
        cells = []
        for p in POOLS:
            s, f = count[(t, p, True)], count[(t, p, False)]
            cells.append(f"{s}/{f}" if p in ("eval_seen", "unseen", "cv_test") else str(s + f))
        print(f"{t:34s}" + "".join(f"{c:>16s}" for c in cells))
    tot = Counter((r["pool"], r["success"]) for r in rows)
    print("total: " + ", ".join(f"{p} {tot[(p, True)]} succ / {tot[(p, False)]} fail" for p in POOLS))
    print(f"wrote {out / 'index.json'} ({len(rows)} episodes)")


if __name__ == "__main__":
    main()
