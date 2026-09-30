"""Does the Q-ensemble detect failed VLA rollouts? Headline numbers from saved score dumps.

Reads one or more dumps of ``experiments/eval_real_robot.py`` (one per checkpoint: zero-shot
Bridge/Fractal ensemble and finetuned ones) and, per channel, writes:

* pre-registered: AUROC(test failures vs test successes) of the episode max, +- bootstrap std,
  and the same restricted to within-task pairs (task-confound control)
* deployment: episode-level conformal detector at ``--alpha`` calibrated on the disjoint
  calibration successes -> false-alarm rate on test successes, recall on test failures, median
  detection time, and the AUROC of the time-normalised score it thresholds
* positive control, if ``--swap`` dumps are given: same episodes, another task's instruction

Writes results.json and results.md into ``--out_dir``. Invocations: docs/REPRODUCE.md, section 1c / 3b.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from say_no import vla_rollouts as vr


def parse_labelled(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        label, sep, path = it.partition("=")
        if not sep:
            raise ValueError(f"expected label=path, got {it!r}")
        out[label] = path
    return out


def markdown(res: dict) -> str:
    lines = [f"# VLA rollout failure detection -- {res['dataset']}", "",
             f"rendered {res['rendered']} by src/scripts/analyse_vla_rollout_failure_detection.py",
             f"alpha={res['alpha']}, split_frac={res['split_frac']}, "
             f"{res['n_split_seeds']} calibration splits, {res['n_boot']} bootstrap resamples, "
             f"horizon {res['horizon_arg']}", "",
             "| checkpoint | frames scored | AUROC of full episode length | AUROC of scored length "
             "| calib: r(mean Q, discounted step count) | RMSE | mean Q range |",
             "|---|---|---|---|---|---|---|",
             *[f"| {k} | {d['horizon'] or 'all'} | {d['episode_length_auroc']:.3f} | "
               f"{d['truncated_length_auroc']:.3f} | {d['critic_fit_calib']['pearson_r']:.3f} | "
               f"{d['critic_fit_calib']['rmse']:.2f} | {d['critic_fit_calib']['q_range'][0]:.1f}.."
               f"{d['critic_fit_calib']['q_range'][1]:.1f} |" for k, d in res["checkpoints"].items()], "",
             "## Pre-registered: AUROC of the episode max, test failures vs test successes", "",
             "| checkpoint | step | channel | AUROC | within-task AUROC | n succ / fail |",
             "|---|---|---|---|---|---|"]
    for label, d in res["checkpoints"].items():
        for c, v in d["channels"].items():
            a, w = v["raw_max"], v["raw_max_within_task"]
            lines.append(f"| {label} | {d['step']} | {c} | {a['auroc']:.3f} +- {a['std']:.3f} | "
                         f"{w['auroc']:.3f} +- {w['std']:.3f} | {a['n_neg']} / {a['n_pos']} |")
    lines += ["", f"## Conformal detector at alpha={res['alpha']} (calibrated on successes only)", "",
              "| checkpoint | channels | false-alarm rate [95% CI] | recall [95% CI] | "
              "median detection time (frac. of episode) | time-normalised AUROC | "
              "time-normalised within-task AUROC |",
              "|---|---|---|---|---|---|---|"]
    for label, d in res["checkpoints"].items():
        for name, op in d["operating_points"].items():
            flo, fhi = op["false_alarm_ci95"]
            rlo, rhi = op["recall_ci95"]
            dt = "--" if op["detection_time_median"] is None else f"{op['detection_time_median']:.2f}"
            lines.append(f"| {label} | {name} | {op['false_alarm_rate']:.3f} [{flo:.2f}, {fhi:.2f}] | "
                         f"{op['recall']:.3f} [{rlo:.2f}, {rhi:.2f}] | {dt} | "
                         f"{op['time_normalised_auroc']:.3f} | "
                         f"{op['time_normalised_within_task_auroc']:.3f} |")
    lines += ["", "## SAFE protocol (gu2025): ROC-AUC = max up to the task's shortest rollout; functional "
              "CP 30/70, np.quantile; T-det over all failures, misses = 1", "",
              "| checkpoint | score | SAFE ROC-AUC | alpha | TPR | TNR | bal-acc | T-det | T-det (detected) | TWA |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for label, d in res["checkpoints"].items():
        for name, sp in d["safe_protocol"].items():
            for a in ("0.1", "0.15"):
                m = sp["by_alpha"][a]
                tp = "--" if m["t_det_tp"] is None else f"{m['t_det_tp']:.3f}"
                lines.append(f"| {label} | {name} | {sp['roc_auc']:.3f} | {a} | {m['tpr']:.3f} | "
                             f"{m['tnr']:.3f} | {m['bal_acc']:.3f} | {m['t_det']:.3f} | {tp} | "
                             f"{m['twa']:.3f} |")
    if res["swap_controls"]:
        lines += ["", "## Positive control: same test episodes, another task's instruction", "",
                  "| checkpoint | channel | n | swapped > own (max) | paired AUROC, max [95% CI] | "
                  "swapped > own (mean) | paired AUROC, mean [95% CI] |",
                  "|---|---|---|---|---|---|---|"]
        for label, cs in res["swap_controls"].items():
            for c in cs:
                lo, hi = c["paired_auroc_ci95"]
                mlo, mhi = c["mean_paired_auroc_ci95"]
                lines.append(f"| {label} | {c['channel']} | {c['n']} | {c['frac_swapped_higher']:.2f} | "
                             f"{c['paired_auroc']:.3f} [{lo:.2f}, {hi:.2f}] | "
                             f"{c['mean_frac_swapped_higher']:.2f} | "
                             f"{c['mean_paired_auroc']:.3f} [{mlo:.2f}, {mhi:.2f}] |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="name printed in the report")
    ap.add_argument("--dump", nargs="+", required=True,
                    help="label=path of each eval_real_robot.py dump (pools calib+test)")
    ap.add_argument("--swap", nargs="*", default=[],
                    help="label=path of swapped-instruction dumps; label must match a --dump")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--split_frac", type=float, default=0.5)
    ap.add_argument("--n_split_seeds", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--policy_csv_dir", default=None,
                    help="SAFE's unzipped openvla_widowx dir: adds the policy's token entropy "
                         "as a baseline channel")
    ap.add_argument("--actions_npz", default=None,
                    help="raw actions per episode (experiments/export_cache_actions.py): adds the "
                         "action-extremeness baseline channel")
    ap.add_argument("--discount", type=float, required=True, help="the finetune run's gamma")
    ap.add_argument("--num_terminal", type=int, required=True,
                    help="the finetune run's terminal frames per success (critic-fit gate)")
    ap.add_argument("--test_tasks", default="",
                    help="comma-separated tasks: restrict the test set to them (seen/unseen breakdown)")
    ap.add_argument("--view", default=None,
                    help="calib_pool:test_pool for SAFE-protocol dumps (pools holdout / eval_seen "
                         "/ unseen), e.g. eval_seen:unseen; omit for calib+test dumps")
    ap.add_argument("--horizon", default="none",
                    help="'none', 'min_calib' (shortest calibration success) or a frame count: "
                         "score only each episode's first frames (see vla_rollouts.truncate)")
    args = ap.parse_args()

    view = tuple(args.view.split(":")) if args.view else None
    dumps = {k: vr.load_dump(p, view) for k, p in parse_labelled(args.dump).items()}
    if args.policy_csv_dir:
        for d in dumps.values():
            vr.attach_policy_entropy(d["records"], args.policy_csv_dir)
    if args.actions_npz:
        for d in dumps.values():
            vr.attach_action_extremeness(d["records"], args.actions_npz, d["action_statistics"])
    res = dict(dataset=args.dataset, rendered=datetime.now().isoformat(timespec="seconds"),
               alpha=args.alpha, split_frac=args.split_frac, n_split_seeds=args.n_split_seeds,
               horizon_arg=args.horizon, view=args.view, policy_csv_dir=args.policy_csv_dir,
               actions_npz=args.actions_npz,
               n_boot=args.n_boot, dumps=parse_labelled(args.dump),
               checkpoints={k: vr.analyse_dump(d, args.alpha, args.split_frac,
                                               args.n_split_seeds, args.n_boot, args.horizon,
                                               args.discount, args.num_terminal,
                                               tuple(t for t in args.test_tasks.split(",") if t))
                            for k, d in dumps.items()},
               swap_controls={})
    for label, path in parse_labelled(args.swap).items():
        swapped = vr.load_dump(path)
        if not swapped["swap_instruction"]:
            raise ValueError(f"{path} was not scored with --swap_instruction")
        res["swap_controls"][label] = [vr.swap_control(dumps[label], swapped, c)
                                       for c in vr.OUR_CHANNELS]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    md = markdown(res)
    (out / "results.md").write_text(md)
    print(md)
    print(f"wrote {out / 'results.json'}, {out / 'results.md'}")


if __name__ == "__main__":
    main()
