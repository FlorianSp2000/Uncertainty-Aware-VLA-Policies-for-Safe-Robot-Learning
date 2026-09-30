"""Figure: balanced accuracy against detection time across the conformal level alpha
(SAFE, gu2025, Fig. 4), one panel per dataset.

Reads results.json files of ``analyse_vla_rollout_failure_detection.py`` (their
``safe_protocol`` section) at one checkpoint each. One point per alpha; T-det = first alarm step
over episode length, averaged over all failed test rollouts with misses counted as 1 (SAFE §5.4).
Up and left is better. The alpha = 0.15 point (SAFE operating point) is ringed, with its min-max range over the
calibration splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from say_no.utils.figstyle import THESIS, save_figure, style  # noqa: E402
from say_no.utils.palette import COLORS  # noqa: E402

SAFE_ALPHA = str(0.15)  # SAFE's operating point, ringed
SERIES = [  # (safe_protocol key, checkpoint role, label, colour, line style, marker)
    ("value_score/step", "primary", r"Value score $-\bar Q_t$, finetuned", COLORS["value_score"], "-", "s"),
    ("value_score/cumsum", "primary", r"Value score $-\bar Q_t$, cumulative, finetuned", COLORS["value_score"], "--", "s"),
    ("ensemble_disagreement/step", "primary", r"Ensemble disagreement $\sigma_{Q,t}$, finetuned", COLORS["ensemble_disagreement"], "-", "o"),
    ("value_score/step", "zeroshot", r"Value score $-\bar Q_t$, before finetuning", COLORS["value_score"], ":", "^"),
    ("policy_token_entropy/step", "primary", "Policy token entropy (baseline)", COLORS["neutral_dark"], "-", "x"),
    ("action_extremeness/step", "primary", "Action extremeness (baseline)", COLORS["neutral_marker"], "-", "+"),
    ("gripper_open_steps/step", "primary", "Gripper-open steps (baseline)", COLORS["neutral_dark"], "--", "d"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--primary", nargs="+", required=True, help="finetuned checkpoint label per file")
    ap.add_argument("--titles", nargs="+", required=True, help="panel title per file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = [json.loads(Path(p).read_text()) for p in args.results]
    ncol = min(len(res), 2)
    nrow = -(-len(res) // ncol)
    with plt.rc_context(style(THESIS, width=1.0, height=2.0 * nrow + 1.0)):
        fig, axes = plt.subplots(nrow, ncol, squeeze=False, layout="constrained", sharey=True)
        for ax, r, prim, title in zip(axes.flat, res, args.primary, args.titles):
            for key, role, label, colour, ls, mk in SERIES:
                d = r["checkpoints"]["zeroshot" if role == "zeroshot" else prim]
                if key not in d["safe_protocol"]:
                    continue
                by = d["safe_protocol"][key]["by_alpha"]
                al = sorted(by, key=float)
                ax.plot([by[a]["t_det"] for a in al], [by[a]["bal_acc"] for a in al], color=colour,
                        ls=ls, marker=mk, ms=3, lw=1.0, label=label)
                if SAFE_ALPHA in by:
                    m = by[SAFE_ALPHA]
                    lo, hi = m["bal_acc_seed_range"]
                    # Range over the calibration splits: how much one draw of the band moves it.
                    ax.errorbar(m["t_det"], m["bal_acc"], yerr=[[m["bal_acc"] - lo], [hi - m["bal_acc"]]],
                                fmt="o", ms=7, mfc="none", color=colour, lw=0.8, capsize=2)
            ax.axhline(0.5, color=COLORS["chance"], lw=0.7, ls=(0, (2, 3)))
            ax.set_title(title, loc="left")
        for ax in axes[-1]:
            ax.set_xlabel(r"Detection time $T_\text{det}$ (fraction of episode)")
        fig.supylabel("Balanced accuracy", fontsize=plt.rcParams["axes.labelsize"])
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
        prov = ", ".join(f"{Path(p).parent.name}:{k}" for p, k in zip(args.results, args.primary))
        save_figure(fig, args.out, f"{prov} | ringed alpha 0.15 | plot_vla_rollout_tradeoff.py",
                    preview=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
