"""Figure: failure-detection quality across finetuning under SAFE's seen / unseen-task protocol,
mean +- std over the folds, one row per rollout dataset.

Reads ``<root>/<dataset>/fold<k>/<view>/results.json`` written by
``run_vla_rollouts_safe_protocol.py``. Columns: unseen tasks (view A, SAFE's: calibrated on
eval-seen successes) and seen tasks (view B: calibrated on held-out training successes). y =
within-task AUROC of the time-normalised episode score (the statistic the conformal detector
thresholds); step 0 = Bridge/Fractal ensemble before finetuning. Flat lines: critic-free
baselines, mean over folds of the last checkpoint's analysis (they do not depend on the critic).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from say_no.utils.figstyle import THESIS, save_figure, style  # noqa: E402
from say_no.utils.palette import COLORS  # noqa: E402

VIEWS = {"B_holdout-cal_evalseen-test": "seen tasks", "A_evalseen-cal_unseen-test": "unseen tasks"}
DATASET_NAMES = {"safe_widowx": "SAFE-WidowX", "libero_pi0fast": "LIBERO-10"}
CHANNELS = {"value_score": dict(marker="s", label=r"$-\bar Q$ (ours)"),
            "ensemble_disagreement": dict(marker="o", label=r"$\sigma_Q$ (ours)")}
BASELINES = {"gripper_open_steps": ("gripper-open steps", COLORS["neutral_dark"], (0, (5, 2))),
             "action_extremeness": ("action extremeness", COLORS["neutral_marker"], (0, (4, 1, 1, 1)))}


def step_of(label: str) -> int:
    return 0 if label == "zeroshot" else int(label.removeprefix("step"))


def within(d: dict, channel: str) -> float:
    """Time-normalised within-task AUROC; the cumulative gripper count has no time-normalised form
    (constant across episodes at t = 0), so its raw max = final count is used."""
    if channel in d["operating_points"]:
        return d["operating_points"][channel]["time_normalised_within_task_auroc"]
    return d["channels"][channel]["raw_max_within_task"]["auroc"]


def panel(ax, results: list[dict]) -> None:
    labels = sorted(set.intersection(*(set(r["checkpoints"]) for r in results)), key=step_of)
    steps = np.array([step_of(k) for k in labels]) / 1000
    for c, st in CHANNELS.items():
        v = np.array([[within(r["checkpoints"][k], c) for k in labels] for r in results])
        m, s = v.mean(0), v.std(0, ddof=1)
        ax.plot(steps, m, color=COLORS[c], marker=st["marker"], ms=3.2, lw=1.2, label=st["label"])
        ax.fill_between(steps, m - s, m + s, color=COLORS[c], alpha=0.18, lw=0)
    last = labels[-1]
    for b, (name, colour, ls) in BASELINES.items():
        vals = [within(r["checkpoints"][last], b) for r in results
                if b in r["checkpoints"][last]["operating_points"] or b in r["checkpoints"][last]["channels"]]
        if vals:
            ax.axhline(np.mean(vals), color=colour, ls=ls, lw=1.0, label=f"{name} (baseline)")
    ax.axhline(0.5, color=COLORS["chance"], lw=0.8, ls=(0, (2, 3)), label="chance")
    ax.set_xlabel("Ensemble finetuning step (thousands)")
    ax.set_ylabel("Within-task AUROC")
    ax.set_ylim(0.3, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    ap.add_argument("--out", required=True, help="PDF path")
    args = ap.parse_args()
    root = Path(args.root)
    with plt.rc_context(style(THESIS, width=1.0, height=1.9 * len(args.datasets) + 0.6)):
        fig, axes = plt.subplots(len(args.datasets), len(VIEWS), squeeze=False, sharex=True, sharey=True,
                                 layout="constrained")
        for row, ds in enumerate(args.datasets):
            folds = sorted((root / ds).glob("fold*"))
            for col, (view, title) in enumerate(VIEWS.items()):
                results = [json.loads((f / view / "results.json").read_text()) for f in folds]
                panel(axes[row, col], results)
                axes[row, col].set_title(f"{DATASET_NAMES[ds]}, {title}", loc="left")
                if col:
                    axes[row, col].set_ylabel("")
                if row < len(args.datasets) - 1:  # shared x axis: label the bottom row only
                    axes[row, col].set_xlabel("")
        h, l = axes[0, 0].get_legend_handles_labels()
        h.append(plt.Rectangle((0, 0), 1, 1, color=COLORS["neutral_marker"], alpha=0.25, lw=0))
        l.append("±1 std over folds")
        fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
        save_figure(fig, args.out, f"{root} ({', '.join(args.datasets)}; mean ± std over folds) | "
                                   "src/scripts/plot_vla_rollout_safe_protocol_curves.py", preview=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
