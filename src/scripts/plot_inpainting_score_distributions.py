"""Ensemble-disagreement distributions on the object-removal evaluation, original vs perturbed scene.

Input, one of:
- `--dump`: the canonical perturbation dump of `eval_perturbation_sensitivity.py` (the 258 reviewed
  pairs; sigma_Q with ddof = 1 as in every thesis table) -- the thesis figure;
- `--hdf5` + `--pkl`: the Sept-2025 per-trajectory dump of
  ``experiments/utils/evaluate_inpainting_uncertainty.py`` (293 rows, sigma_Q with ddof = 0) and the
  reviewed pkl whose ``failed_ids`` drop the rejected pairs -- an earlier draft figure.
`--profile` picks the figstyle target: thesis (exact size, provenance in the PDF metadata) or icra (tight bbox, no
stamp, bytes pinned). Invocation: src/scripts/render_thesis_figures.py.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from say_no.cp.bands import split_cp_threshold
from say_no.perturbation.load import load_models
from say_no.perturbation.scores import member_q, spread
from say_no.utils.figstyle import PROFILES, save_figure, save_pinned, style
from say_no.utils.palette import COLORS


def load_reviewed_rows(hdf5_path: Path, pkl_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with h5py.File(hdf5_path) as f:
        meta = json.loads(f.attrs["metadata"])
        ids = f["trajectory_ids"][()]
        orig = f["original_uncertainties"][()]
        inp = f["inpainted_uncertainties"][()]
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    trajs = [t for t in data["trajectories"]
             if t.get("first_image_inpainted") is not None and t.get("last_image_inpainted") is not None]
    if len(trajs) != len(ids) or not np.array_equal(ids, np.array([t["trajectory_id"] for t in trajs])):
        raise ValueError("HDF5 rows and pkl trajectories do not line up; refusing to guess the join")
    failed = {(f["dataset"], f["trajectory_id"]) for f in data["failed_ids"]}
    keep = np.array([(t["dataset"], t["trajectory_id"]) not in failed for t in trajs])
    return orig[keep], inp[keep], meta["model_id"]


def load_canonical(dump: Path) -> tuple[np.ndarray, np.ndarray, str]:
    m = load_models([(dump.stem, dump)])[dump.stem]
    if m.kind != "ensemble" or "inpaint" not in m.conditions:
        raise ValueError(f"{dump}: expected an ensemble dump with the object-removal condition")
    return spread(member_q(m.q["orig"], "both")), spread(member_q(m.q["inpaint"], "both")), dump.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dump", type=Path, help="canonical perturbation dump (thesis)")
    src.add_argument("--hdf5", type=Path, help="Sept-2025 inpainting dump (ICRA); needs --pkl")
    ap.add_argument("--pkl", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--width", required=True, type=float, help="fraction of the profile's \\columnwidth")
    ap.add_argument("--height", type=float, default=2.1, help="inches")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--threshold_line", action="store_true",
                    help="dashed line at the mean over originals (post hoc; ICRA draft only)")
    ap.add_argument("--conformal_alpha", type=float, default=None,
                    help="dashed line at the split-conformal threshold of the originals at this level (thesis)")
    args = ap.parse_args()
    if (args.hdf5 is None) != (args.pkl is None):
        raise ValueError("--pkl goes with --hdf5 and only with it")
    if args.threshold_line and args.conformal_alpha is not None:
        raise ValueError("--threshold_line and --conformal_alpha draw competing thresholds; pick one")

    orig, inp, source = load_canonical(args.dump) if args.dump else load_reviewed_rows(args.hdf5, args.pkl)
    n = len(orig)
    print(f"n={n}  mean shift={float((inp - orig).mean()):+.3f}  %increase={float((inp > orig).mean()):.3f}")

    edges = np.linspace(0.0, float(max(orig.max(), inp.max())), args.bins + 1)
    profile = PROFILES[args.profile]
    with plt.rc_context(style(profile, width=args.width, height=args.height)):
        fig, ax = plt.subplots()
        ax.hist(orig, bins=edges, alpha=0.75, color=COLORS["original"], label="Original")
        ax.hist(inp, bins=edges, histtype="step", linewidth=1.2, color=COLORS["perturbed"], label="Object removal")
        if args.threshold_line:
            ax.axvline(float(orig.mean()), color=COLORS["aggregate"], linestyle="--", linewidth=0.8,
                       label="Mean over originals")
        if args.conformal_alpha is not None:
            # illustration on all originals; the table's detection rate uses random calibration halves
            tau = split_cp_threshold(orig, args.conformal_alpha)
            print(f"conformal threshold (alpha={args.conformal_alpha:g}, all originals): {tau:.3f}")
            ax.axvline(tau, color=COLORS["threshold"], linestyle="--", linewidth=0.9,
                       label=rf"Threshold $\tau$, $\alpha = {args.conformal_alpha:g}$")
        # thesis: sentence-case axis labels like every other thesis figure; the ICRA draft keeps its pinned labels
        thesis = args.profile == "thesis"
        ax.set_xlabel(r"Ensemble disagreement $\sigma_Q$" if thesis else r"Ensemble Disagreement $\sigma_Q$")
        ax.set_ylabel("Number of trajectories" if thesis else "Number of Trajectories")
        ax.legend(frameon=False, handlelength=1.2, handletextpad=0.5, borderaxespad=0.2)
        if args.profile == "thesis":
            save_figure(fig, args.out, f"{source} · n={n} · {Path(__file__).name}")
        else:  # double-anonymous submission: no stamp
            save_pinned(fig, args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
