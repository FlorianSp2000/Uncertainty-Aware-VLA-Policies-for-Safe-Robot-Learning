"""Thesis tables of the VLA-rollout failure-detection experiment, SAFE protocol (3 folds, step 50k).

Reads the per-fold results.json files of analyse_vla_rollout_failure_detection.py (written by
run_vla_rollouts_safe_protocol.py) and the per-fold split indices of split_vla_rollouts.py; writes
four LaTeX tables into --out_dir:

* vla_safe_detection.tex    -- SAFE-WidowX, seen (view B) vs unseen tasks (view A), every detector
* vla_libero_detection.tex  -- LIBERO-10 pi0-FAST, same layout, no token-entropy row
* vla_safe_folds.tex        -- SAFE-WidowX tasks and per-fold split sizes (appendix)
* vla_safe_per_fold.tex     -- SAFE-WidowX per fold, seen/unseen, SayNo vs gripper (appendix)

Numbers equal the 3-fold blocks of summary_<dataset>_<view>_<checkpoint>.md
(summarise_vla_rollout_results.py). Every path is a CLI argument; the invocation is written into
each table's header comment.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

VIEW_SEEN = "B_holdout-cal_evalseen-test"  # calibrate on holdout successes, test eval-seen rollouts
VIEW_UNSEEN = "A_evalseen-cal_unseen-test"  # SAFE's protocol: calibrate on eval-seen successes
FOLDS = (0, 1, 2)
DETECTORS = [  # key in safe_protocol, channel key, row label
    ("value_score/step", "value_score", r"SayNo value score $-\bar Q_t$"),
    ("ensemble_disagreement/step", "ensemble_disagreement", r"SayNo ensemble disagreement $\sigma_{Q,t}$"),
    ("gripper_open_steps/step", "gripper_open_steps", "Gripper-open steps"),
    ("action_extremeness/step", "action_extremeness", "Action extremeness"),
    ("policy_token_entropy/step", "policy_token_entropy", "OpenVLA token entropy"),
]
# column: (label, getter, higher is better, ranked). Recall, FAR and T_det are not ranked: a detector
# that alarms on almost every rollout wins recall and T_det, one that never alarms wins FAR
COLS = [("Recall", lambda m, w: m["tpr"], True, False),
        ("FAR", lambda m, w: 1 - m["tnr"], False, False),
        ("Bal.\\ Acc.", lambda m, w: m["bal_acc"], True, True),
        (r"$T_\text{det}$", lambda m, w: m["t_det"], False, False),
        ("AUROC", lambda m, w: w, True, True)]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _results(root: Path, dataset: str, view: str) -> list[dict]:
    return [_load(root / dataset / f"fold{k}" / view / "results.json") for k in FOLDS]


def _metrics(res: dict, ckpt: str, alpha: str, sp_key: str, ch: str) -> tuple[dict, float]:
    """SAFE-protocol metrics at alpha (mean over all calibration draws) and within-task AUROC
    (time-normalised episode max; raw final count for channels without an operating point)."""
    d = res["checkpoints"][ckpt]
    m = d["safe_protocol"][sp_key]["by_alpha"][alpha]
    w = (d["operating_points"][ch]["time_normalised_within_task_auroc"] if ch in d["operating_points"]
         else d["channels"][ch]["raw_max_within_task"]["auroc"])
    return m, w


def _rank_marks(values: list[float], higher: bool) -> list[str]:
    """bold best, underline second best (thesis table convention); ranked on the printed value."""
    r = [round(v, 2) for v in values]
    order = sorted(set(r), reverse=higher)
    return ["best" if v == order[0] else "second" if len(order) > 1 and v == order[1] else "" for v in r]


def _fmt(v: float, pm: float | None, mark: str) -> str:
    s = f"{v:.2f}"
    s = r"\mathbf{" + s + "}" if mark == "best" else r"\underline{" + s + "}" if mark == "second" else s
    return f"${s}" + (r"{\scriptstyle\,\pm\,%.2f}$" % pm if pm is not None else "$")


def _mean_std(v: list[float]) -> tuple[float, float]:
    if len(v) < 2:
        raise ValueError(f"need >= 2 folds for a std, got {len(v)}")
    return statistics.fmean(v), statistics.stdev(v)  # stdev = ddof 1


def detection_table(root: Path, dataset: str, detectors: list, ckpt: str, alpha: str,
                    caption: str, label: str) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}l" + "c" * len(COLS) + "@{}}", r"\toprule",
             "Detector & " + " & ".join(c[0] for c in COLS) + r" \\", r"\midrule"]
    for gi, (gname, view) in enumerate([("Seen tasks", VIEW_SEEN), ("Unseen tasks", VIEW_UNSEEN)]):
        res = _results(root, dataset, view)
        rows = []
        for sp_key, ch, lab in detectors:
            per = [_metrics(r, ckpt, alpha, sp_key, ch) for r in res]
            rows.append((lab, [_mean_std([get(m, w) for m, w in per]) for _, get, _, _ in COLS]))
        marks = [_rank_marks([r[1][j][0] for r in rows], COLS[j][2]) if COLS[j][3] else [""] * len(rows)
                 for j in range(len(COLS))]
        if gi:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (len(COLS) + 1, gname))
        for i, (lab, cells) in enumerate(rows):
            lines.append(lab + " & " + " & ".join(_fmt(v, pm, marks[j][i])
                                                  for j, (v, pm) in enumerate(cells)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}", r"\label{" + label + "}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def _indices(paths: list[str]) -> list[list[dict]]:
    if len(paths) != len(FOLDS):
        raise ValueError(f"expected {len(FOLDS)} fold indices, got {len(paths)}")
    return [json.loads(Path(p).read_text()) for p in paths]


def _unseen_tasks(idx: list[dict]) -> list[str]:
    return sorted({r["condition"] for r in idx if r["pool"] == "unseen"})


def _span(v: list[int]) -> str:
    return str(v[0]) if min(v) == max(v) else f"{min(v)}--{max(v)}"


def folds_table(paths: list[str]) -> str:
    idx = _indices(paths)
    instr = {r["condition"]: r["instruction"] for r in idx[0]}
    if any({r["key"] for r in i} != {r["key"] for r in idx[0]} for i in idx):
        raise ValueError("fold indices do not hold the same rollouts")
    held = {t: k for k, i in zip(FOLDS, idx) for t in _unseen_tasks(i)}
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Task & Succ.\,/\,Fail. & Unseen Fold & Finetuning & Holdout & Test (Succ.\,/\,Fail.) \\",
             r"\midrule"]
    for t in sorted(instr, key=lambda t: instr[t]):
        c0 = Counter(r["success"] for r in idx[0] if r["condition"] == t)
        seen = [Counter((r["pool"], r["success"]) for r in i if r["condition"] == t)
                for k, i in zip(FOLDS, idx) if held.get(t) != k]
        get = lambda pool, ok: _span([c[(pool, ok)] for c in seen])
        lines.append(f"{instr[t]} & {c0[True]}\\,/\\,{c0[False]} & {held.get(t, '--')} & "
                     f"{get('finetune', True)} & {get('holdout', True)} & "
                     f"{get('eval_seen', True)}\\,/\\,{get('eval_seen', False)} \\\\")
    tot = [Counter((r["pool"], r["success"]) for r in i) for i in idx]
    gt = lambda pool, ok: _span([c[(pool, ok)] for c in tot])
    n = Counter(r["success"] for r in idx[0])
    lines += [r"\midrule",
              f"Total & {n[True]}\\,/\\,{n[False]} & & {gt('finetune', True)} & {gt('holdout', True)} & "
              f"{gt('eval_seen', True)}\\,/\\,{gt('eval_seen', False)} \\\\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{Tasks of the OpenVLA WidowX rollouts released with SAFE~\cite{gu2025} and their "
              r"split in the three folds. Each fold withholds two tasks entirely (unseen tasks); rollouts "
              r"of the other six tasks are split into finetuning successes, held-out calibration successes "
              r"and the seen-task test set; the remaining failures of seen tasks are not used. Split columns count the folds "
              r"in which the task is seen; a range means the count differs between those folds.}",
              r"\label{tab:vla-safe-folds}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def per_fold_table(root: Path, paths: list[str], ckpt: str, alpha: str) -> str:
    idx = _indices(paths)
    instr = {r["condition"]: r["instruction"] for r in idx[0]}
    dets = DETECTORS[:3]
    short = [r"$-\bar Q_t$", r"$\sigma_{Q,t}$", "Gripper"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}p{4.2cm}l" + "c" * 2 * len(dets) + "@{}}", r"\toprule",
             r" & & \multicolumn{%d}{c}{Bal.\ Acc.} & \multicolumn{%d}{c}{AUROC} \\" % (len(dets), len(dets)),
             r"\cmidrule(lr){3-%d}\cmidrule(l){%d-%d}" % (2 + len(dets), 3 + len(dets), 2 + 2 * len(dets)),
             "Fold (Unseen Tasks) & Test Tasks & " + " & ".join(short * 2) + r" \\", r"\midrule"]
    views = [("Seen", _results(root, "safe_widowx", VIEW_SEEN)), ("Unseen", _results(root, "safe_widowx", VIEW_UNSEEN))]
    for k, i in zip(FOLDS, idx):
        unseen = _unseen_tasks(i)
        if unseen != sorted(views[1][1][k]["checkpoints"][ckpt]["tasks"]):
            raise ValueError(f"fold {k}: index unseen tasks {unseen} != results tasks")
        if k:
            lines.append(r"\midrule")
        for vi, (vname, res) in enumerate(views):
            per = [_metrics(res[k], ckpt, alpha, sp, ch) for sp, ch, _ in dets]
            first = f"{k} ({instr[unseen[0]]}; {instr[unseen[1]]})" if vi == 0 else ""
            lines.append(first + f" & {vname} & " + " & ".join(
                [_fmt(m["bal_acc"], None, "") for m, _ in per] + [_fmt(w, None, "") for _, w in per]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Every SAFE-WidowX fold separately, same protocol and checkpoint as "
              r"Table~\ref{tab:vla-safe-detection} ($\alpha = 0.15$, 50k finetuning steps, balanced "
              r"accuracy averaged over 10 calibration splits). Seen: test rollouts of the six "
              r"finetuning tasks, calibrated on held-out successes; unseen: rollouts of the two withheld "
              r"tasks, calibrated on successful test rollouts of seen tasks. AUROC within task; for "
              r"the gripper the final open-step count.}",
              r"\label{tab:vla-safe-per-fold}", r"\end{table}"]
    return "\n".join(lines) + "\n"


SAFE_CAPTION = (
    r"Failure detection by SayNo and three baselines on OpenVLA rollouts on a real WidowX (SAFE "
    r"release~\cite{gu2025}), for seen and unseen tasks. Three folds each withhold two "
    r"of eight tasks from finetuning and calibration. Seen tasks: test rollouts of the six other tasks, "
    r"calibrated on 30 held-out successes never used for finetuning; unseen tasks: rollouts of the two "
    r"withheld tasks, calibrated on the 60 successful test rollouts of seen tasks (SAFE's protocol). "
    r"Every detector alarms when its score leaves the band fitted on the calibration successes "
    r"($\alpha = 0.15$); the critic is finetuned for 50k steps on 93 successes. FAR: false-alarm rate; "
    r"$T_\text{det}$: first alarm as a fraction of the episode, a missed failure counted as 1; AUROC: "
    r"within-task, threshold-free (time-normalised episode maximum; for gripper-open steps the final count). "
    r"Means over 10 calibration splits, then mean $\pm$ standard deviation over the 3 folds. Only balanced "
    r"accuracy and AUROC are ranked.")
LIBERO_CAPTION = (
    r"Failure detection by SayNo and two baselines on $\pi_0$-FAST rollouts on LIBERO-10 collected by "
    r"Foresight~\cite{zhang2026}, for seen and unseen tasks. Three folds each withhold three of ten tasks "
    r"from finetuning and calibration; seen and unseen tasks as in Table~\ref{tab:vla-safe-detection}, "
    r"calibrated on 30--37 held-out successes (seen) or on the 82--99 successful test rollouts of "
    r"seen tasks (unseen). Episodes are cut at each task's shortest "
    r"calibration success (unseen tasks: the shortest overall), since every failure is a timeout and length "
    r"alone would separate the classes. The critic is finetuned for 50k steps on 94--112 successes; "
    r"$\alpha = 0.15$; means over 10 calibration splits, then mean $\pm$ standard deviation over the 3 "
    r"folds. Only balanced accuracy and AUROC are ranked.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="src/results/vla-rollout-failure-detection/safe_protocol")
    ap.add_argument("--safe_index", nargs=3, required=True, help="SAFE-WidowX split index of folds 0, 1, 2")
    ap.add_argument("--checkpoint", required=True, help="checkpoint key, pre-registered: step50000")
    ap.add_argument("--alpha", required=True, help="key of safe_protocol.by_alpha, e.g. 0.15")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    header = ("% {name} -- generated by src/scripts/vla_rollout_thesis_tables.py; do not edit; invocation: "
              + " ".join(sys.argv[1:]).replace("\\", "/") + "\n")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck, a = args.checkpoint, args.alpha
    tables = {
        "vla_safe_detection.tex": detection_table(root, "safe_widowx", DETECTORS, ck, a, SAFE_CAPTION,
                                                  "tab:vla-safe-detection"),
        "vla_libero_detection.tex": detection_table(root, "libero_pi0fast", DETECTORS[:4], ck, a,
                                                    LIBERO_CAPTION, "tab:vla-libero-detection"),
        "vla_safe_folds.tex": folds_table(args.safe_index),
        "vla_safe_per_fold.tex": per_fold_table(root, args.safe_index, ck, a),
    }
    for name, tex in tables.items():
        (out / name).write_text(header.format(name=name) + tex, encoding="utf-8")
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
