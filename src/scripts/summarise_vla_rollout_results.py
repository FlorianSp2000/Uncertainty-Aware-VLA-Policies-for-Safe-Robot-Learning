"""One table of the headline failure-detection metrics across datasets and test sets.

Reads results.json files written by ``analyse_vla_rollout_failure_detection.py`` and prints /
writes a markdown table: per (test set, score) the SAFE-protocol metrics at one alpha -- recall
(TPR), false-alarm rate (1 - TNR), balanced accuracy, T-det (all failures, misses = 1), T-det
over detected failures, TWA -- plus SAFE's ROC-AUC and our time-normalised within-task AUROC.
Summary for inspection; the thesis tables come from vla_rollout_thesis_tables.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCORES = [("value_score/step", "value_score", "−Q̄ (ours)"),
          ("ensemble_disagreement/step", "ensemble_disagreement", "σ_Q (ours)"),
          ("gripper_open_steps/step", "gripper_open_steps", "gripper-open steps (baseline)"),
          ("action_extremeness/step", "action_extremeness", "action extremeness (baseline)"),
          ("policy_token_entropy/step", "policy_token_entropy", "policy token entropy (baseline)")]


def rows(label: str, res: dict, ckpt: str, alpha: str) -> list[dict]:
    d = res["checkpoints"][ckpt]
    out = []
    for sp_key, ch, name in SCORES:
        if sp_key not in d["safe_protocol"]:
            continue
        sp = d["safe_protocol"][sp_key]
        m = sp["by_alpha"][alpha]
        within = (d["operating_points"][ch]["time_normalised_within_task_auroc"]
                  if ch in d["operating_points"] else d["channels"][ch]["raw_max_within_task"]["auroc"])
        out.append(dict(test_set=label, n=f"{d['n_test_success']}/{d['n_test_failure']}", score=name,
                        recall=m["tpr"], far=1 - m["tnr"], bal_acc=m["bal_acc"],
                        bal_acc_range=m["bal_acc_seed_range"], t_det=m["t_det"],
                        t_det_tp=m["t_det_tp"], twa=m["twa"], safe_roc=sp["roc_auc"],
                        within_auroc=within))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", nargs="+", required=True, help="label=results.json:checkpoint")
    ap.add_argument("--alpha", default="0.15")
    ap.add_argument("--group", nargs="*", default=[],
                    help="name=label1,label2,...: also report the mean [min-max] over these test sets")
    ap.add_argument("--out", required=True, help="markdown path")
    args = ap.parse_args()
    table = []
    for e in args.entry:
        label, _, rest = e.partition("=")
        path, _, ckpt = rest.rpartition(":")
        if not (label and path and ckpt):
            raise ValueError(f"expected label=path:checkpoint, got {e!r}")
        table += rows(label, json.loads(Path(path).read_text()), ckpt, args.alpha)
    lines = [f"SAFE protocol at alpha = {args.alpha} (functional CP on calibration successes, 30/70, "
             "10 calibration splits averaged); within-task AUROC = our time-normalised episode max "
             "(raw final count for the gripper baseline)", "",
             "| test set | succ/fail | score | recall | false-alarm rate | bal-acc [split range] | "
             "T-det (misses=1) | T-det detected | TWA | SAFE ROC-AUC | within-task AUROC |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        tp = "--" if r["t_det_tp"] is None else f"{r['t_det_tp']:.2f}"
        lo, hi = r["bal_acc_range"]
        lines.append(f"| {r['test_set']} | {r['n']} | {r['score']} | {r['recall']:.2f} | {r['far']:.2f} | "
                     f"{r['bal_acc']:.3f} [{lo:.2f}, {hi:.2f}] | {r['t_det']:.2f} | {tp} | {r['twa']:.2f} | "
                     f"{r['safe_roc']:.3f} | {r['within_auroc']:.3f} |")
    for g in args.group:
        name, _, labels = g.partition("=")
        members = labels.split(";")
        missing = [m for m in members if m not in {r["test_set"] for r in table}]
        if missing:
            raise ValueError(f"group {name}: unknown test sets {missing}")
        lines += ["", f"### {name}: mean ± std (ddof 1) [min-max] over {len(members)} test sets", "",
                  "| score | recall | false-alarm rate | bal-acc | T-det (misses=1) | T-det detected | "
                  "within-task AUROC | SAFE ROC-AUC |", "|---|---|---|---|---|---|---|---|"]
        for _, _, score in SCORES:
            rs = [r for r in table if r["test_set"] in members and r["score"] == score]
            if len(rs) != len(members):
                continue

            def agg(k):
                v = np.array([r[k] for r in rs if r[k] is not None])
                return f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.2f} [{v.min():.2f}-{v.max():.2f}]"
            lines.append(f"| {score} | {agg('recall')} | {agg('far')} | {agg('bal_acc')} | "
                         f"{agg('t_det')} | {agg('t_det_tp')} | {agg('within_auroc')} | "
                         f"{agg('safe_roc')} |")
    md = "\n".join(lines) + "\n"
    Path(args.out).write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
