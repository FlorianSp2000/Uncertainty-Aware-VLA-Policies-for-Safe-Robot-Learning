"""Does the disagreement score react to a WRONG instruction, or only to an UNFAMILIAR one?

Consumes the dumps written by external/V-GPS/experiments/eval_iva_three_way_prompt.py, which score
every eval false-premise-split frame under three kinds of instruction while holding image,
action and timestep fixed:

    true      the episode's own instruction
    inj       false premise injection prompt -- a novel composition, never seen in training
    other_j   the train-modal instruction of source task j -- a real string seen thousands
              of times in training, wrong for this scene

Every AUROC below uses the SAME negatives (`true` on feasible frames), so the only thing
that changes between them is which wrong string was substituted. `AUROC_inj` reproduces the
construction behind the project's published IVA numbers; `AUROC_other_inj` is the same
measurement on the same frames with a swapped task prompt instead of a novel one.
The difference between them is the result.

"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from say_no.cp import _compat
from say_no.cp.episode_eval import MIN_CALIB_LEN, contrast, episode_scores, operating_point
from say_no.cp.metrics import auroc


def collect(dump_path):
    """-> per-task pooled arrays, the raw records by task, and the dump header."""
    dump = _compat.load(str(dump_path))
    by_task = defaultdict(lambda: defaultdict(list))
    records_by_task = defaultdict(list)
    foreign_order = {}
    for rec in dump["trajectories"]:
        task = rec["task"]
        records_by_task[task].append(rec)
        s_true, s_inj, s_other, injected = episode_scores(rec)
        feasible = ~injected
        foreign_order[task] = rec["foreign_tasks"]
        # The contrastive score, on the same three conditions. It is a candidate DETECTOR, so
        # it has to carry the same hard-negative control as the raw score, not just an AUROC.
        c_true = contrast(s_true, s_other)
        c_inj = contrast(s_inj, s_other)
        c_other = np.stack([contrast(s_other[j], s_other, exclude=j)
                            for j in range(s_other.shape[0])], 0)
        acc = by_task[task]
        acc["c_true_feasible"].append(c_true[feasible])
        acc["c_other_feasible"].append(c_other[:, feasible])
        acc["true_feasible"].append(s_true[feasible])
        acc["other_feasible"].append(s_other[:, feasible])
        acc["q_true_feasible"].append(rec["q_members_true"].astype(np.float64).mean(0)[feasible])
        if injected.any():
            acc["true_at_inj"].append(s_true[injected])
            acc["inj_at_inj"].append(s_inj[injected])
            acc["other_at_inj"].append(s_other[:, injected])
            acc["c_inj_at_inj"].append(c_inj[injected])
            acc["c_other_at_inj"].append(c_other[:, injected])
            acc["q_true_at_inj"].append(rec["q_members_true"].astype(np.float64).mean(0)[injected])
            acc["q_inj_at_inj"].append(rec["q_members_given"].astype(np.float64).mean(0)[injected])
            acc["q_other_at_inj"].append(rec["q_members_foreign"].astype(np.float64).mean(1)[:, injected])
    out = {}
    for task, acc in by_task.items():
        out[task] = {
            k: (np.concatenate(v, axis=-1) if v[0].ndim > 1 else np.concatenate(v))
            for k, v in acc.items()
        }
        out[task]["foreign_tasks"] = foreign_order[task]
    return out, dict(records_by_task), dump


def _auroc_against(negatives, positives):
    labels = np.concatenate([np.zeros(negatives.size, bool), np.ones(positives.size, bool)])
    return float(auroc(labels, np.concatenate([negatives, positives])))


def task_report(d):
    """Every number for one task. Negatives are always `true` on feasible frames."""
    neg = d["true_feasible"]
    row = {
        "n_feasible_frames": int(neg.size),
        "n_injected_frames": int(d["inj_at_inj"].size),
        "floor": float(neg.mean()),
        "auroc_inj": _auroc_against(neg, d["inj_at_inj"]),
        "auroc_other_inj": _auroc_against(neg, d["other_at_inj"].ravel()),
        "auroc_other_all": _auroc_against(neg, d["other_feasible"].ravel()),
        # Same three comparisons under the contrastive score.
        "contrast_auroc_inj": _auroc_against(d["c_true_feasible"], d["c_inj_at_inj"]),
        "contrast_auroc_other_inj": _auroc_against(d["c_true_feasible"],
                                                   d["c_other_at_inj"].ravel()),
        "contrast_auroc_other_all": _auroc_against(d["c_true_feasible"],
                                                   d["c_other_feasible"].ravel()),
        # DIAGNOSTIC: paired against the same frame under its true instruction. Needs ground
        # truth, so it is not detector performance -- it isolates the language effect.
        "rise_inj": float((d["inj_at_inj"] - d["true_at_inj"]).mean()),
        "rise_other_at_inj": float((d["other_at_inj"] - d["true_at_inj"][None, :]).mean()),
        "rise_other_all": float((d["other_feasible"] - d["true_feasible"][None, :]).mean()),
        "frac_positive_inj": float((d["inj_at_inj"] > d["true_at_inj"]).mean()),
        "frac_positive_other": float((d["other_feasible"] > d["true_feasible"][None, :]).mean()),
        "q_true_feasible": float(d["q_true_feasible"].mean()),
        "q_shift_inj": float((d["q_inj_at_inj"] - d["q_true_at_inj"]).mean()),
        "q_shift_other": float((d["q_other_at_inj"] - d["q_true_at_inj"][None, :]).mean()),
    }
    row["auroc_drop"] = row["auroc_inj"] - row["auroc_other_inj"]
    # Which source task supplied the string, so a null cannot hide behind one odd source.
    row["auroc_by_source"] = {
        src: _auroc_against(neg, d["other_at_inj"][j])
        for j, src in enumerate(d["foreign_tasks"])
    }
    return row


def summarise(per_task):
    keys = ["auroc_inj", "auroc_other_inj", "auroc_other_all", "auroc_drop", "floor",
            "rise_inj", "rise_other_at_inj", "rise_other_all", "q_shift_inj", "q_shift_other",
            "contrast_auroc_inj", "contrast_auroc_other_inj", "contrast_auroc_other_all"]
    return {f"mean_{k}": float(np.mean([r[k] for r in per_task.values()])) for k in keys}


def figure(models, path, stamp=""):
    """Per-task AUROC under the two wrong-prompt regimes, one point per task per checkpoint.

    Legend carries series identity only, never results.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = ["o", "s", "^", "D", "v", "P"]
    colours = ["#4c72b0", "#c44e52", "#55a868", "#8172b2", "#ccb974", "#64b5cd"]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    # Limits come from the data. A fixed floor silently drops any checkpoint whose
    # hard-negative AUROC is below it -- which is exactly the checkpoint worth seeing.
    values = [r[k] for m in models.values() for r in m["per_task"].values()
              for k in ("auroc_inj", "auroc_other_inj")]
    lo = min(0.45, min(values) - 0.04)
    lim = (lo, 1.02)
    ax.plot(lim, lim, color="0.4", lw=1, ls="--", zorder=1)
    ax.axhline(0.5, color="0.75", lw=0.8, ls=":", zorder=0)
    ax.axvline(0.5, color="0.75", lw=0.8, ls=":", zorder=0)

    for i, (label, m) in enumerate(models.items()):
        x = [r["auroc_inj"] for r in m["per_task"].values()]
        y = [r["auroc_other_inj"] for r in m["per_task"].values()]
        s = m["summary"]
        ax.scatter(x, y, marker=markers[i % len(markers)], s=42, alpha=0.65,
                   color=colours[i % len(colours)], edgecolor="none", zorder=2, label=label)
        ax.scatter([s["mean_auroc_inj"]], [s["mean_auroc_other_inj"]],
                   marker=markers[i % len(markers)], s=190, zorder=3,
                   facecolor=colours[i % len(colours)], edgecolor="black", linewidth=1.4)

    ax.set_xlabel("AUROC against false premise injection prompt")
    ax.set_ylabel("AUROC against a swapped task prompt")
    ax.set_title("Q ensemble AUROC on injected and swapped prompts, per IVA task", fontsize=10)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    # Lower left: the only quadrant no checkpoint occupies (it is below the diagonal AND
    # below chance on the benchmark stimulus).
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.95,
              title="small: one task | large: mean", title_fontsize=7.5)
    mid = lim[0] + 0.42 * (lim[1] - lim[0])
    ax.annotate("y = x", (mid, mid), textcoords="offset points", xytext=(7, -11),
                fontsize=7.5, color="0.45")
    fig.tight_layout()
    if stamp:
        fig.text(0.004, 0.004, stamp, fontsize=4.6, color="0.45", ha="left", va="bottom")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True, metavar="LABEL=PKL",
                    help="repeatable; one three-way-prompt pickle per checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--stamp", default="", help="provenance footer drawn on the figure")
    ap.add_argument("--alphas", default="0.10,0.20")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--align", default="truncate", choices=("truncate", "pad"))
    ap.add_argument("--min-calib-len", type=int, default=MIN_CALIB_LEN)
    ap.add_argument("--pool-tasks", action="store_true",
                    help="also calibrate the conformal quantile across tasks, which is what "
                         "makes levels finer than 1/11 attainable")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    models = {}
    for spec in a.dump:
        label, path = spec.split("=", 1)
        by_task, records_by_task, dump = collect(path)
        per_task = {t: task_report(d) for t, d in by_task.items()}
        ops = [operating_point(records_by_task, alpha=al, statistic=st, k=k, seeds=seeds,
                               score=sc, frame_statistic=fs, pool_tasks=pool,
                               align=a.align, min_calib_len=a.min_calib_len)
               for al in [float(x) for x in a.alphas.split(",")]
               for st, k in [("max", 0), ("topk_mean", 5)]
               for fs in ("disagreement", "neg_mean_q", "both")
               for sc in ("raw", "contrast")
               for pool in ((False, True) if a.pool_tasks else (False,))]
        models[label] = {
            "pkl": path, "step": dump["step"], "model_dir": dump["model_dir"],
            "n_members": len(dump["member_ids"]),
            "instruction_split": dump["instruction_split"],
            "representatives": dump["representatives"],
            "per_task": per_task, "summary": summarise(per_task),
            "operating_points": [o for o in ops if o is not None],
        }

    for label, m in models.items():
        print(f"\n=== {label}  (step {m['step']}, {m['n_members']} members) ===")
        print(f"{'task':30s} {'floor':>7s} {'AUROC_inj':>10s} {'AUROC_other':>12s} "
              f"{'drop':>7s} {'rise_inj':>9s} {'rise_other':>11s}")
        for t in sorted(m["per_task"], key=lambda t: -m["per_task"][t]["auroc_inj"]):
            r = m["per_task"][t]
            print(f"{t:30s} {r['floor']:7.3f} {r['auroc_inj']:10.3f} {r['auroc_other_inj']:12.3f} "
                  f"{r['auroc_drop']:7.3f} {r['rise_inj']:9.3f} {r['rise_other_at_inj']:11.3f}")
        s = m["summary"]
        print(f"{'MEAN':30s} {s['mean_floor']:7.3f} {s['mean_auroc_inj']:10.3f} "
              f"{s['mean_auroc_other_inj']:12.3f} {s['mean_auroc_drop']:7.3f} "
              f"{s['mean_rise_inj']:9.3f} {s['mean_rise_other_at_inj']:11.3f}")
        print(f"{'':30s} AUROC with a swapped task prompt on ALL frames: "
              f"{s['mean_auroc_other_all']:.3f}")
        print(f"{'':30s} contrastive score  AUROC_inj {s['mean_contrast_auroc_inj']:.3f}  "
              f"AUROC_other {s['mean_contrast_auroc_other_inj']:.3f}  "
              f"AUROC_other(all frames) {s['mean_contrast_auroc_other_all']:.3f}")
        print(f"{'':30s} mean Q shift: injected {s['mean_q_shift_inj']:+.2f}, "
              f"swapped {s['mean_q_shift_other']:+.2f}")
        if m["operating_points"]:
            print(f"  episode-level detector ({len(seeds)} seeds), recall on the two "
                  f"wrong-instruction regimes:")
            print(f"   {'alpha':>6s} {'frame score':>14s} {'pool':>9s} {'stat':>12s} {'xtask':>5s} "
                  f"{'ep FAR':>21s} {'recall: IVA inj':>21s} {'recall: swapped':>21s}")
            for o in sorted(m["operating_points"],
                            key=lambda o: -min(o["episode_recall_iva_injection"],
                                               o["episode_recall_foreign_instruction"])):
                def ci(v, c):
                    return f"{v:6.3f} [{c[0]:.3f},{c[1]:.3f}]"
                print(f"   {o['alpha']:6.2f} {o['frame_statistic']:>14s} {o['score']:>9s} "
                      f"{o['statistic']:>12s} {str(o['pool_tasks']):>5s} "
                      f"{ci(o['episode_far'], o['far_ci']):>21s} "
                      f"{ci(o['episode_recall_iva_injection'], o['recall_iva_ci']):>21s} "
                      f"{ci(o['episode_recall_foreign_instruction'], o['recall_foreign_ci']):>21s}")
            best = min(m["operating_points"],
                       key=lambda o: -min(o["episode_recall_iva_injection"],
                                          o["episode_recall_foreign_instruction"]))
            worst = best["per_task"]
            print(f"   per task at the best row ({best['frame_statistic']}/{best['score']}/"
                  f"{best['statistic']}): worst IVA-injection recall "
                  f"{min(v['recall_iva'] for v in worst.values()):.3f} "
                  f"({min(worst, key=lambda t: worst[t]['recall_iva'])}), "
                  f"worst swapped-prompt recall "
                  f"{min(v['recall_foreign'] for v in worst.values()):.3f} "
                  f"({min(worst, key=lambda t: worst[t]['recall_foreign'])})")

    if a.fig:
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        figure(models, a.fig, a.stamp)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(models, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
