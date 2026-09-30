"""Figure: how early the online detector fires -- fraction of failed rollouts detected, and of
successful rollouts falsely alarmed, by each point of the episode (SAFE, gu2025, Fig. 10 style).

One panel per dump. SAFE's functional CP band at ``--alpha``, calibrated on the dump's
calibration successes, averaged over calibration splits. Scores: our value score and ensemble
disagreement, and the gripper-open-steps baseline. Alarms use only the prefix up to each step
(``vla_rollouts.safe_alarm_curves``); episodes are not padded past their own end.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from say_no import vla_rollouts as vr  # noqa: E402
from say_no.utils.figstyle import THESIS, save_figure, style  # noqa: E402
from say_no.utils.palette import COLORS  # noqa: E402

SCORES = [("value_score", r"value score $-\bar Q_t$", COLORS["value_score"], "s"),
          ("ensemble_disagreement", r"ensemble disagreement $\sigma_{Q,t}$", COLORS["ensemble_disagreement"], "o"),
          ("gripper_open_steps", "gripper-open steps", COLORS["neutral_dark"], "d")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--titles", nargs="+", required=True)
    ap.add_argument("--actions_npz", nargs="+", required=True, help="one per dump")
    ap.add_argument("--test_tasks", nargs="+", default=None,
                    help="per dump: comma-separated tasks to restrict the test set to, or 'all'")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tt = args.test_tasks or ["all"] * len(args.dumps)
    if not len(args.dumps) == len(args.titles) == len(args.actions_npz) == len(tt):
        raise ValueError("one title, actions file and test-task spec per dump")

    n = len(args.dumps)
    ncol = min(n, 2)
    nrow = -(-n // ncol)
    with plt.rc_context(style(THESIS, width=1.0, height=2.0 * nrow + 1.0)):
        fig, axes = plt.subplots(nrow, ncol, squeeze=False, layout="constrained", sharey=True,
                                 sharex=True)
        for ax, path, title, acts, tasks in zip(axes.flat, args.dumps, args.titles, args.actions_npz, tt):
            dump = vr.load_dump(path)
            vr.attach_action_extremeness(dump["records"], acts, dump["action_statistics"])
            calib, succ, fail = vr.populations(dump["records"])
            if tasks != "all":
                keep = set(tasks.split(","))
                succ = [r for r in succ if r["condition"] in keep]
                fail = [r for r in fail if r["condition"] in keep]
            for key, label, colour, mk in SCORES:
                c = vr.safe_alarm_curves(calib, succ, fail, vr.CHANNELS[key], args.alpha, args.n_seeds)
                ax.plot(c["t_frac"], c["failures_detected"], color=colour, lw=1.3,
                        label=f"Recall, {label}")
                ax.plot(c["t_frac"], c["successes_alarmed"], color=colour, lw=1.0, ls=":",
                        label=f"False-alarm rate, {label}")
            ax.set_title(f"{title}\n({len(succ)} successes / {len(fail)} failures)", loc="left")
            ax.set_ylim(0, 1)
        for ax in axes[-1]:
            ax.set_xlabel("Fraction of episode elapsed")
        fig.supylabel(rf"Share of rollouts with alarm ($\alpha = {args.alpha:g}$)", fontsize=plt.rcParams["axes.labelsize"])
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
        prov = ", ".join(Path(p).stem.replace("_calib-test", "") for p in args.dumps)
        save_figure(fig, args.out, f"{prov} | plot_vla_rollout_detection_time.py",
                    preview=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
