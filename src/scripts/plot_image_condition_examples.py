"""Thesis figure: a few reviewed trajectories under object removal and its three controls (original,
object removal, edit-region control, background control, re-encoding control). Pixels rebuilt with
say_no.perturbation.images.rebuild_like_dump, which asserts they equal what the cluster eval scored.
Invocation: src/scripts/render_thesis_figures.py."""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from say_no.perturbation.images import rebuild_like_dump
from say_no.perturbation.load import load_models, reviewed_frames
from say_no.perturbation.names import condition_name
from say_no.utils.figstyle import FULL, THESIS, save_figure, style

FRAME_INDEX = {"first": 0, "last": 1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--hdf5", required=True, type=Path)
    ap.add_argument("--radius", required=True, type=int)
    ap.add_argument("--frame", required=True, choices=sorted(FRAME_INDEX))
    ap.add_argument("--examples", required=True, nargs="+", help="dataset:trajectory_id")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    dump = load_models([("ref", args.hdf5)])["ref"]
    trajs = reviewed_frames(args.pkl, dump.trajectory_ids, dump.dataset)
    images, info = rebuild_like_dump(trajs, args.hdf5)
    columns = [("Original", None)] + [(condition_name(c), c) for c in
                                      ("inpaint", f"composite_local_r{args.radius}",
                                       f"composite_global_r{args.radius}", "jpeg_orig")]
    for _, c in columns:
        if c is not None and c not in images:
            raise ValueError(f"condition {c} not in {args.hdf5}")
    key = {(str(t["dataset"]), int(t["trajectory_id"])): i for i, t in enumerate(trajs)}
    rows = []
    for e in args.examples:
        ds, tid = e.rsplit(":", 1)
        if (ds, int(tid)) not in key:
            raise ValueError(f"{e} not among the reviewed rows")
        rows.append(key[(ds, int(tid))])

    f = FRAME_INDEX[args.frame]
    rc = style(THESIS, width=FULL, height=1.05 * len(rows) + 0.35)
    # manual layout: the row labels sit in a fixed left margin, the images keep their aspect
    with plt.rc_context({**rc, "figure.constrained_layout.use": False}):
        fig, axes = plt.subplots(len(rows), len(columns), squeeze=False)
        for r, i in enumerate(rows):
            for c, (name, cond) in enumerate(columns):
                ax = axes[r][c]
                img = trajs[i]["first_image" if f == 0 else "last_image"] if cond is None else images[cond][i, f]
                ax.imshow(img)
                ax.set_xticks([]), ax.set_yticks([])
                if r == 0:
                    ax.set_title(name.replace(" ", "\n", 1))
            instruction = str(trajs[i]["language"]).strip()
            axes[r][0].set_ylabel("\n".join(textwrap.wrap(f"“{instruction}”", 17)),
                                  fontsize=plt.rcParams["xtick.labelsize"])
        fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.14, right=0.995, top=0.88, bottom=0.04)
        save_figure(fig, args.out, f"{args.hdf5.name}, {args.frame} frame, mask radius {args.radius} px, "
                                   f"JPEG q{info['jpeg_quality']} · plot_image_condition_examples.py", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
