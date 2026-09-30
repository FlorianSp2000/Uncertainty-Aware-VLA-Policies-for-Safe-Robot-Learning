"""Bridge/Fractal training vs validation instructions, and detection AUROC split by seen / unseen instruction.

Input: the full instruction walk (slurm/extract-instruction-vocabulary.sbatch) and the
perturbation-sensitivity dumps. Output: one JSON with the vocabulary overlap per source dataset, the
split check of the evaluation pkls, and per model the AUROC table (say_no.perturbation.evaluate) on the
rows whose instruction is / is not in the training split; plus the three-row comparison of the thesis table
(training-split trajectories, held-out trajectories with a training instruction, held-out trajectories
with an instruction not in the training split) and that table. Library: say_no.perturbation.instructions.
Invocation with the paths used: docs/REPRODUCE.md, step 2c.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from say_no.perturbation.evaluate import condition_frame_table
from say_no.perturbation.instructions import NORMALISATIONS, check_pkl_split, load_walk, overlap_counts, seen_mask
from say_no.perturbation.load import load_models, restrict
from say_no.perturbation.thesis_tables import seen_unseen_domains

SET_CONDITIONS = {"val200": ["lang_swap_train", "lang_swap_near", "lang_swap_pkl"],
                  "inpainting258": ["lang_swap_train", "inpaint"]}
DOMAIN_CONDITIONS = ["lang_swap_train", "inpaint"]  # random relabeling, object removal


def labelled_paths(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for it in items:
        lab, p = it.rsplit("=", 1)
        if not Path(p).exists():
            raise FileNotFoundError(p)
        out.append((lab, Path(p)))
    return out


def stratified(model, mask_by_group: dict[str, np.ndarray], conditions, args) -> dict:
    out = {}
    for group, mask in mask_by_group.items():
        n = int(mask.sum())
        if n < args.min_rows:
            out[group] = {"n": n, "auroc": f"not computed: fewer than {args.min_rows} rows"}
            continue
        out[group] = {"n": n, "auroc": condition_frame_table(restrict(model, mask), args.n_boot, args.seed,
                                                             args.n_folds, conditions)}
    return out


def domain_rows(train, held_out, walk: dict, args) -> dict:
    """{scope: {group: {n, cells: {cond: {readout: {auroc, ci95}}}}}} at the average of both frames.
    scope 'bridge' compares like with like (unseen instructions exist only on Bridge); 'all' pools Bridge
    and Fractal where the group exists."""
    if not seen_mask(train.language, train.dataset, walk, "exact").all():
        raise ValueError("a training-split trajectory has an instruction outside the training split")
    seen = seen_mask(held_out.language, held_out.dataset, walk, "exact")
    out = {}
    for scope in ("bridge", "all"):
        in_tr = np.ones(len(train.dataset), bool) if scope == "all" else train.dataset == "bridge"
        in_ho = np.ones(len(held_out.dataset), bool) if scope == "all" else held_out.dataset == "bridge"
        groups = {"train": (train, in_tr), "eval_seen": (held_out, in_ho & seen), "eval_unseen": (held_out, in_ho & ~seen)}
        out[scope] = {}
        for g, (m, mask) in groups.items():
            if mask.sum() < args.min_rows:
                raise ValueError(f"{m.label} {scope}/{g}: {mask.sum()} rows < --min_rows")
            t = condition_frame_table(restrict(m, mask), args.n_boot, args.seed, args.n_folds, DOMAIN_CONDITIONS)
            out[scope][g] = {"n": int(mask.sum()),
                             "cells": {c: {rd: t[c]["both"][rd] for rd in ("spread", "neg_mean_q")} for c in DOMAIN_CONDITIONS}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--walk", required=True, type=Path, help="full instruction walk HDF5")
    ap.add_argument("--val", action="append", required=True, metavar="LABEL=HDF5", help="dump on val200")
    ap.add_argument("--inpainting", action="append", required=True, metavar="LABEL=HDF5", help="dump on inpainting258")
    ap.add_argument("--train", action="append", required=True, metavar="LABEL=HDF5",
                    help="dump on the reviewed training-split object-removal trajectories")
    ap.add_argument("--domain_models", nargs="+", required=True, help="labels of the thesis-table rows (SayNo variants)")
    ap.add_argument("--table_out", required=True, type=Path)
    ap.add_argument("--pkl", action="append", required=True, metavar="SPLIT=PKL",
                    help="evaluation pkl and the split it must come from")
    ap.add_argument("--n_boot", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_folds", type=int, required=True)
    ap.add_argument("--min_rows", type=int, required=True, help="smallest group an AUROC is computed on")
    ap.add_argument("--out_json", required=True, type=Path)
    args = ap.parse_args()

    walk = load_walk(args.walk)
    result = {"created_at": datetime.now().isoformat(timespec="seconds"), "walk": str(args.walk),
              "overlap": {how: overlap_counts(walk, how) for how in NORMALISATIONS},
              "pkl_split_check": {}, "sets": {}}
    for split, p in labelled_paths(args.pkl):
        result["pkl_split_check"][p.name] = {"split": split, **check_pkl_split(p, walk, split)}

    for set_name, items in (("val200", args.val), ("inpainting258", args.inpainting)):
        models = load_models(labelled_paths(items))
        ref = next(iter(models.values()))
        entry = {"rows": {}, "models": {}}
        for how in NORMALISATIONS:
            seen = seen_mask(ref.language, ref.dataset, walk, how)
            groups = {f"{d}_{s}": (ref.dataset == d if d != "all" else np.ones_like(seen)) & (seen if s == "seen" else ~seen)
                      for d in ("all", "bridge", "fractal") for s in ("seen", "unseen")}
            entry["rows"][how] = {g: int(m.sum()) for g, m in groups.items()}
            for lab, m in models.items():
                if not (np.array_equal(m.language, ref.language) and np.array_equal(m.dataset, ref.dataset)):
                    raise ValueError(f"{set_name}: {lab} rows differ from {ref.label}")
                entry["models"].setdefault(lab, {})[how] = stratified(m, groups, SET_CONDITIONS[set_name], args)
        result["sets"][set_name] = entry

    train_models, held_out = load_models(labelled_paths(args.train)), load_models(labelled_paths(args.inpainting))
    result["domains"] = {lab: domain_rows(train_models[lab], held_out[lab], walk, args) for lab in args.domain_models}
    seen_unseen_domains(result["domains"], args.domain_models, args.table_out)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
