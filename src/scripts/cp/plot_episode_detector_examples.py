"""One IVA episode scored under three instructions, with the episode-level detector's threshold
and first alarm: own instruction (feasible), IVA's injected prompt, another task's instruction.

Same pixels in all three rows; only the instruction differs. The band is the one that scored
this episode in the evaluation (analysis seed 0, the fold that holds it out), with the cross-task
pooled quantile and the max-over-frames statistic, so nothing drawn is in-sample.

Episode selection is a rule, not a pick: among episodes whose own-instruction scoring does not
alarm, whose injected scoring alarms at or after its first injected frame, and whose first
swapped prompt alarms, the one whose injected-frame count is closest to the dataset median;
ties broken by (task, episode id).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from say_no.cp import _compat
from say_no.cp.bands import _channels, _zscores, first_alarm_frame
from say_no.cp.episode_eval import MIN_CALIB_LEN, fold_bands, prepare
from say_no.utils.figstyle import FULL, THESIS, save_figure, style
from say_no.utils.palette import COLORS

C_SPREAD, C_MEANQ = COLORS["ensemble_disagreement"], COLORS["value_score"]
C_FP = COLORS["false_premise_injection"]
C_THRESH = COLORS["threshold"]
C_PROFILE_END = COLORS["chance"]
# First alarm is an event, not an entity: neutral black, so no palette hue is borrowed.
C_ALARM = "black"
CHANNEL_LABELS = (r"$z_t$ of ensemble disagreement", r"$z_t$ of value score")


LS_MEANQ = (0, (4, 1.5))  # dashed: the two scores stay apart in black-and-white print


def held_out_bands(per_task, *, alpha, seed, n_folds):
    """-> {(task, i): band that scored episode i of task}."""
    out = {}
    for f in range(n_folds):
        bands, holds, _ = fold_bands(per_task, seed=seed, fold=f, n_folds=n_folds, alpha=alpha,
                                     statistic="max", k=0, tau_alpha=None, align="truncate",
                                     min_calib_len=MIN_CALIB_LEN, pool_tasks=True)
        for t, idx in holds.items():
            out.update({(t, i): bands[t] for i in idx})
    return out


def pick(per_task, bands):
    n_inj = [int(m.sum()) for d in per_task.values() for m in d["injected"] if m.any()]
    median = float(np.median(n_inj))
    ok = []
    for t, d in per_task.items():
        for i, sc in enumerate(d["scores"]):
            b, onset = bands[(t, i)], d["onset"][i]
            if onset is None:
                continue
            ch = lambda cond, j=None: [c[cond] if j is None else c[2][j] for c in sc]
            t_inj = first_alarm_frame(ch(1), b)
            if (first_alarm_frame(ch(0), b) is None and t_inj is not None and t_inj >= onset
                    and first_alarm_frame(ch(2, 0), b) is not None):
                ok.append((abs(int(d["injected"][i].sum()) - median), t, d["episode_id"][i], i))
    assert ok, "no episode satisfies the selection rule"
    return min(ok)[1], min(ok)[3], len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    a = ap.parse_args()

    dump = _compat.load(a.dump)
    recs = {}
    for r in dump["trajectories"]:
        recs.setdefault(r["task"], []).append(r)
    per_task = prepare(recs, "both")
    bands = held_out_bands(per_task, alpha=a.alpha, seed=a.seed, n_folds=a.n_folds)
    task, i, n_ok = pick(per_task, bands)
    rec, sc, band = recs[task][i], per_task[task]["scores"][i], bands[(task, i)]
    inj = per_task[task]["injected"][i]
    injected_prompt = next(l for l, m in zip(rec["language"], inj) if m)
    rows = [(0, None, f"True-premise instruction: “{rec['true_instruction']}”"),
            (1, None, f"False-premise injection on {int(inj.sum())} of {inj.size} frames: “{injected_prompt}”"),
            (2, 0, f"Counterfactual relabeling: “{rec['foreign_instructions'][0]}”")]
    zs = [[_zscores(c[cond] if j is None else c[2][j], z["mu"], z["varsigma"])
           for c, z in zip(sc, _channels(band))] for cond, j, _ in rows]
    top = max(float(np.max(z)) for zz in zs for z in zz)
    bottom = min(float(np.min(z)) for zz in zs for z in zz)

    T = inj.size
    t_profile = min(z["T"] for z in _channels(band))
    with plt.rc_context(style(THESIS, width=FULL, height=4.6)):
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
        # the panel titles name the condition; no colour-coded panel edge that the legend would have to explain
        for ax, (cond, j, title), zz in zip(axs, rows, zs):
            if cond == 1:
                for tt in np.flatnonzero(inj):
                    ax.axvspan(tt - 0.5, tt + 0.5, color=C_FP, alpha=0.15, lw=0)
            ax.plot(zz[0], color=C_SPREAD, lw=1.1)
            ax.plot(zz[1], color=C_MEANQ, lw=1.1, ls=LS_MEANQ)
            ax.axhline(band["q_M"], color=C_THRESH, lw=1.1, ls="--")
            # past the shortest calibration episode the feasible profile is flat-extended
            ax.axvline(t_profile - 0.5, color=C_PROFILE_END, lw=0.8, ls=":")
            t_al = first_alarm_frame([c[cond] if j is None else c[2][j] for c in sc], band)
            if t_al is not None:
                zmax = max(zz[0][t_al], zz[1][t_al])
                ax.plot([t_al], [zmax], marker="v", color=C_ALARM, ms=7, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.6, clip_on=False)
            ax.set_yscale("symlog", linthresh=10)
            ax.set_ylim(bottom - 1, top * 1.6)
            ax.set_title(title, loc="left")
        fig.supylabel("Time-normalized score $z_t$", fontsize=plt.rcParams["axes.labelsize"])
        axs[-1].set_xlabel("Step $t$")
        axs[-1].set_xlim(-0.5, T - 0.5)
        handles = [Line2D([], [], color=C_SPREAD, lw=1.2), Line2D([], [], color=C_MEANQ, lw=1.2, ls=LS_MEANQ),
                   Line2D([], [], color=C_THRESH, lw=1.2, ls="--"),
                   Patch(color=C_FP, alpha=0.15),
                   Line2D([], [], marker="v", color=C_ALARM, ls="none", ms=6,
                          markeredgecolor="white", markeredgewidth=0.6),
                   Line2D([], [], color=C_PROFILE_END, lw=0.8, ls=":")]
        labels = [*CHANNEL_LABELS, f"Threshold $\\tau$ ($\\alpha = {a.alpha:g}$)",
                  "False-premise (FP) frame", "First alarm", "End of reference time profile"]
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.025))
        fig.get_layout_engine().set(rect=(0, 0.125, 1, 0.875))
        provenance = (f"{Path(a.dump).name} | {task} episode {rec['episode_id']} | seed {a.seed}, "
                      f"held-out fold, pooled quantile | {n_ok} episodes met the selection rule | "
                      f"{Path(__file__).name}")
        save_figure(fig, a.out, provenance, preview=True)
    print(f"{task} episode {rec['episode_id']}: {int(inj.sum())} injected frames, onset "
          f"{per_task[task]['onset'][i]}, q_M {band['q_M']:.3f}; {n_ok} candidates. wrote {a.out}")


if __name__ == "__main__":
    main()
