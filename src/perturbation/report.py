"""One call from dumps to the results dict; qualitative example rows. Tables: say_no.perturbation.thesis_tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from say_no.perturbation.evaluate import (condition_frame_table, member_subsets, operating_point,
                                          paired_scores, per_frame_shift, split_half_operating_point)
from say_no.perturbation.load import ModelDump, restrict
from say_no.perturbation.scores import crossfit_folds, member_q, plain_readouts, spread
from say_no.perturbation.tables import tex_escape, write_booktabs


def op_readouts(m: ModelDump) -> tuple[str, ...]:
    return ("spread", "neg_mean_q") if m.kind == "ensemble" else (m.readout,)


def calibration_scores(train: ModelDump) -> dict[str, np.ndarray]:
    """Scores of training-split originals (frame 'both') that calibrate an ensemble's threshold."""
    if train.kind != "ensemble" or train.conditions != ["orig"]:
        raise ValueError(f"{train.label}: expected an ensemble dump of originals only")
    r = plain_readouts(train.q["orig"], "both")
    return {"spread": r["spread"], "neg_mean_q": r["neg_mean_q"]}


def analyse(models: dict[str, ModelDump], train: dict[str, ModelDump], *, n_boot: int, seed: int, n_folds: int,
            alphas: list[float], n_splits: int, op_condition: str, member_sizes: list[int], n_subsets: int,
            subset_conditions: list[str], dataset_conditions: list[str]) -> dict:
    """Per model: AUROC [paired CI] per condition x frame x readout (all rows and per source dataset
    for `dataset_conditions`), member-subset curves, split-half and training-split operating points."""
    out = {}
    for lab, m in models.items():
        r = {"kind": m.kind, "readout": m.readout, "paths": m.paths, "n_trajectories": int(len(m.trajectory_ids)),
             "n_by_dataset": {d: int((m.dataset == d).sum()) for d in np.unique(m.dataset)},
             "auroc": condition_frame_table(m, n_boot, seed, n_folds)}
        r["auroc_by_dataset"] = {d: condition_frame_table(restrict(m, m.dataset == d), n_boot, seed, n_folds,
                                                          conditions=dataset_conditions)
                                 for d in np.unique(m.dataset)}
        if m.kind == "ensemble":
            r["n_members"] = m.n_members
            r["per_frame_shift"] = per_frame_shift(m)
            r["member_subsets"] = {c: member_subsets(m, c, [k for k in member_sizes if k <= m.n_members], n_subsets, seed)
                                   for c in subset_conditions if c in m.conditions}
        if op_condition in m.conditions:
            folds = crossfit_folds(len(m.trajectory_ids), n_folds, seed)
            s = paired_scores(m, op_condition, "both", folds)
            op = {}
            for rd in op_readouts(m):
                neg, pos = s[rd]
                op[rd] = {"split_half_test_originals": {str(a): split_half_operating_point(neg, pos, a, n_splits, seed)
                                                        for a in alphas}}
                if lab in train:
                    cal = calibration_scores(train[lab])[rd]
                    op[rd]["training_split_originals"] = {str(a): operating_point(cal, neg, pos, a) for a in alphas}
            r["operating_point"] = {"condition": op_condition, "frame": "both", "readouts": op}
        out[lab] = r
    return out


def pick_examples(model: ModelDump, cond: str, quantiles: list[float]) -> list[int]:
    """Trajectories whose paired change of the spread (both frames) under `cond` sits closest
    to each quantile: shows a success, a typical case and a failure, not a hand pick."""
    d = spread(member_q(model.q[cond], "both")) - spread(member_q(model.q["orig"], "both"))
    order = np.argsort(d)
    return [int(order[min(len(d) - 1, int(round(q * (len(d) - 1))))]) for q in quantiles]


def example_rows(model: ModelDump, idx: list[int], swap_cond: str) -> list[dict]:
    rows = []
    for i in idx:
        r = {"row": i, "trajectory_id": int(model.trajectory_ids[i]), "dataset": str(model.dataset[i]),
             "instruction": str(model.language[i]), "swapped_instruction": str(model.swap_language[swap_cond][i])}
        for c in ("orig", swap_cond, "inpaint"):
            r[c] = {"mean_q_first": float(member_q(model.q[c][i:i + 1], "first").mean()),
                    "mean_q_last": float(member_q(model.q[c][i:i + 1], "last").mean()),
                    "spread_both": float(spread(member_q(model.q[c][i:i + 1], "both"))[0]),
                    "member_q_first": member_q(model.q[c][i:i + 1], "first")[0].tolist(),
                    "member_q_last": member_q(model.q[c][i:i + 1], "last")[0].tolist()}
        rows.append(r)
    return rows


def write_example_table(rows: list[dict], swap_cond: str, path: Path) -> None:
    body = []
    for r in rows:
        body.append([r["dataset"], tex_escape(r["instruction"]), tex_escape(r["swapped_instruction"])] + sum(
            ([f"{r[c]['mean_q_first']:.1f}", f"{r[c]['mean_q_last']:.1f}", f"{r[c]['spread_both']:.2f}"]
             for c in ("orig", swap_cond, "inpaint")), []))
    sub = ["$\\bar Q$ first", "$\\bar Q$ last", "$\\sigma_Q$"]
    write_booktabs(path, ["data", "instruction", "swapped instr."] + sub * 3, body,
                   align="lp{2.6cm}p{2.6cm}" + "c" * 9,
                   group_header=[("", 3), ("original", 3), ("swapped instr.", 3), ("inpainted", 3)])
