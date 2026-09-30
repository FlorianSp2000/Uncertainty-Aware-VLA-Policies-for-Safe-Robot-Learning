"""Figure: failure-detection AUROC across finetuning, and the per-step scores behind it, one row
per rollout dataset.

Reads the results.json written by ``analyse_vla_rollout_failure_detection.py`` (one per
dataset) and the dump of the checkpoint given by ``--primary``. Columns:

1. within-task AUROC (test failures vs test successes of the same task, pair-weighted over tasks)
   per finetuning checkpoint; "base" = the Bridge/Fractal ensemble before finetuning. Raw max =
   the pre-registered episode max of the frame score; time-normalised = the statistic the
   conformal detector thresholds. Flat lines: critic-free baselines.
2. / 3. time-normalised value score and ensemble disagreement per rollout step at that checkpoint:
   median of test failures (band: interquartile range) and of test successes (dotted: quartiles).
   Both channels in every row, so the rows compare the same quantities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from say_no import vla_rollouts as vr  # noqa: E402
from say_no.cp import bands  # noqa: E402
from say_no.utils.figstyle import THESIS, save_figure, style  # noqa: E402
from say_no.utils.palette import COLORS  # noqa: E402

CHANNEL_STYLE = {"value_score": dict(marker="s", symbol=r"$-\bar Q$", name="value score"),
                 "ensemble_disagreement": dict(marker="o", symbol=r"$\sigma_Q$",
                                               name="ensemble disagreement")}
BASELINES = {"policy_token_entropy": ("policy token entropy", COLORS["neutral_dark"], (0, (1, 1))),
             "action_extremeness": ("action extremeness", COLORS["neutral_marker"], (0, (4, 1, 1, 1))),
             "gripper_open_steps": ("gripper-open steps", COLORS["neutral_dark"], (0, (5, 2)))}


def _is_base(d: dict) -> bool:
    return "VGPS_ensemble" in d["run_dir"]


def auroc_curve(ax, res: dict) -> None:
    ckpts = sorted(res["checkpoints"].values(), key=lambda d: 0 if _is_base(d) else d["step"])
    steps = [0 if _is_base(d) else d["step"] for d in ckpts]
    for c, st in CHANNEL_STYLE.items():
        raw = np.array([d["channels"][c]["raw_max_within_task"]["auroc"] for d in ckpts])
        z = np.array([d["operating_points"][c]["time_normalised_within_task_auroc"] for d in ckpts])
        ax.plot(steps, raw, color=COLORS[c], ls="--", marker=st["marker"], mfc="white", ms=3.2,
                lw=0.9, label=f"{st['symbol']}, raw max")
        ax.plot(steps, z, color=COLORS[c], ls="-", marker=st["marker"], ms=3.2, lw=1.2,
                label=f"{st['symbol']}, time-normalised")
    # Critic-free baselines do not depend on finetuning: one flat line each, time-normalised.
    # The gripper count is cumulative, so its episode max is its final value; it has no
    # time-normalised form (constant across episodes at t = 0), so its raw max is drawn.
    for b, (name, colour, ls) in BASELINES.items():
        d = ckpts[-1]
        if b in d["operating_points"]:
            v = d["operating_points"][b]["time_normalised_within_task_auroc"]
        elif b in d["channels"]:
            v = d["channels"][b]["raw_max_within_task"]["auroc"]
        else:
            continue
        ax.axhline(v, color=colour, ls=ls, lw=1.0, label=f"{name} (baseline)")
    ax.axhline(0.5, color=COLORS["chance"], lw=0.8, ls=(0, (2, 3)), label="chance")
    ax.set_xticks(steps[:1] + [s for s in steps[1:] if s % max(1, steps[-1] // 5) == 0])
    ax.set_xticklabels(["base"] + [str(s) for s in ax.get_xticks()[1:]])
    ax.set_xlabel("finetuning step")
    ax.set_ylabel("within-task AUROC")
    ax.set_ylim(0.25, 1.0)


def z_profile(ax, res: dict, dump_path: Path, c: str) -> None:
    dump = vr.load_dump(dump_path)
    records, _ = vr.truncate(dump["records"], res["horizon_arg"])
    calib, test_succ, test_fail = vr.populations(records)
    band = bands.episode_cp_threshold_multi([[vr.CHANNELS[c](r) for r in calib]],
                                            alpha=res["alpha"], split_seed=0, align="truncate",
                                            split_frac=res["split_frac"])
    z_of = lambda r: bands._zscores(vr.CHANNELS[c](r), band["channels"][0]["mu"],  # noqa: E731
                                    band["channels"][0]["varsigma"])
    T = min(len(r["s_t"]) for r in calib)
    t = np.arange(T)
    Zf = np.stack([z_of(r)[:T] for r in test_fail if len(r["s_t"]) >= T])
    Zs = np.stack([z_of(r)[:T] for r in test_succ if len(r["s_t"]) >= T])
    ax.plot(t, np.median(Zf, 0), color=COLORS[c], lw=1.3, label="test failures, median (colour = score)")
    ax.fill_between(t, *np.percentile(Zf, [25, 75], 0), color=COLORS[c], alpha=0.2, lw=0,
                    label="test failures, interquartile range")
    ax.plot(t, np.median(Zs, 0), color=COLORS["neutral_dark"], ls="--", lw=1.1,
            label="test successes, median")
    for q in np.percentile(Zs, [25, 75], 0):
        ax.plot(t, q, color=COLORS["neutral_dark"], ls=":", lw=0.7)
    ax.plot([], [], color=COLORS["neutral_dark"], ls=":", lw=0.7, label="test successes, 25th and 75th percentile")
    ax.axhline(0, color=COLORS["chance"], lw=0.6)
    ax.set_xlabel("rollout step")
    ax.set_ylabel(f"time-normalised {CHANNEL_STYLE[c]['symbol']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="results.json per dataset (one row each)")
    ap.add_argument("--primary", nargs="+", required=True, help="checkpoint label per dataset")
    ap.add_argument("--out", required=True, help="PDF path")
    args = ap.parse_args()
    if len(args.results) != len(args.primary):
        raise ValueError("one --primary per --results file")

    results = [json.loads(Path(p).read_text()) for p in args.results]
    n = len(results)
    with plt.rc_context(style(THESIS, width=1.0, height=1.9 * n + 0.9)):
        fig, axes = plt.subplots(n, 3, squeeze=False, layout="constrained",
                                 gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
        for row, (res, prim) in enumerate(zip(results, args.primary)):
            auroc_curve(axes[row, 0], res)
            for col, c in ((1, "value_score"), (2, "ensemble_disagreement")):
                z_profile(axes[row, col], res, Path(res["dumps"][prim]), c)
            axes[row, 0].set_title(res["dataset"], loc="left")
            axes[row, 1].set_title(f"finetuning step {res['checkpoints'][prim]['step']}", loc="left")
        h0, l0 = axes[0, 0].get_legend_handles_labels()
        h1, l1 = axes[0, 1].get_legend_handles_labels()
        fig.legend(h0, l0, loc="outside lower left", ncol=2, frameon=False)
        fig.legend(h1, l1, loc="outside lower right", ncol=1, frameon=False)
        prov = " | ".join(f"{Path(p).parent.parent.name}/{Path(p).parent.name}: {Path(r['dumps'][k]).name}"
                          for p, r, k in zip(args.results, results, args.primary))
        save_figure(fig, args.out, f"{prov} | src/scripts/plot_vla_rollout_failure_detection.py",
                    preview=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
