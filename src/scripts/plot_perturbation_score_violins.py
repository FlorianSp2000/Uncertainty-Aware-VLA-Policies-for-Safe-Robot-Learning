"""Violins of the Bridge/Fractal detector scores as percentiles among the original inputs' scores.

`detectors`: every row of tab:bf-detector-comparison under the three perturbations (replaces the table).
`seen_unseen`: SayNo variants on Bridge training / test-seen-task / test-unseen-task demonstrations.
Library: say_no.perturbation.violins. Invocation: src/scripts/render_thesis_figures.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from say_no.perturbation import violins as V
from say_no.perturbation.instructions import load_walk
from say_no.perturbation.load import load_models
from say_no.perturbation.names import KNN_K


def labelled(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"expected LABEL=VALUE, got {it!r}")
        out.append(tuple(it.rsplit("=", 1)))
    return out


def paths(items: list[str]) -> list[tuple[str, Path]]:
    out = [(lab, Path(p)) for lab, p in labelled(items)]
    for _, p in out:
        if not p.exists():
            raise FileNotFoundError(p)
    return out


def detector_data(args) -> tuple[dict, list[dict]]:
    """(result sets, table rows) of the `detectors` figure; tests/test_perturbation_score_violins.py reuses it."""
    if args.k != KNN_K:
        raise ValueError(f"--k {args.k}: the row names print k-NN of the tables (say_no.perturbation.names.KNN_K = {KNN_K})")
    sets = V.load_detector_sets({"val200": paths(args.val), "inpainting258": paths(args.inpainting)},
                                dict(paths(args.train_ref)), dict(paths(args.clip_scores)), args.clip_label,
                                args.main_ensembles, args.k)
    return sets, V.detector_rows(args.main_ensembles, args.clip_label, dict(labelled(args.color)))


def seen_unseen_groups(args) -> dict[str, dict]:
    """{SayNo variant: {group: ModelDump}} of the `seen_unseen` figure."""
    walk = load_walk(args.walk)
    train, held_out = load_models(paths(args.train)), load_models(paths(args.inpainting))
    return {lab: V.domain_groups(train[lab], held_out[lab], walk) for lab in args.main_ensembles}


def detectors(args) -> None:
    sets, rows = detector_data(args)
    V.detector_figure(rows, V.detector_percentiles(sets, rows), args.out, provenance(args))


def seen_unseen(args) -> None:
    u = {(lab, rd): V.seen_unseen_percentiles(groups, rd)
         for lab, groups in seen_unseen_groups(args).items() for rd in V.SCORE_SYMBOLS}
    V.seen_unseen_figure(u, args.main_ensembles, args.out, provenance(args))


def provenance(args) -> str:
    return f"{Path(__file__).name} {' '.join(sys.argv[1:])}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="figure", required=True)
    a = sub.add_parser("detectors")
    a.add_argument("--val", action="append", required=True, metavar="LABEL=HDF5", help="dump on the relabeling set")
    a.add_argument("--inpainting", action="append", required=True, metavar="LABEL=HDF5", help="dump on the object-removal set")
    a.add_argument("--train_ref", action="append", required=True, metavar="LABEL=HDF5",
                   help="training-split originals with feat/ of each SayNo variant: encoder-novelty reference")
    a.add_argument("--clip_scores", action="append", required=True, metavar="SET=HDF5", help="SET: val200 | inpainting258")
    a.add_argument("--clip_label", required=True)
    a.add_argument("--color", action="append", required=True, metavar="LABEL=KEY",
                   help="say_no.utils.palette key of each SayNo variant and classifier")
    a.add_argument("--k", type=int, required=True, help="k of the encoder-feature kNN")
    b = sub.add_parser("seen_unseen")
    b.add_argument("--walk", required=True, type=Path, help="full instruction walk HDF5")
    b.add_argument("--train", action="append", required=True, metavar="LABEL=HDF5",
                   help="dump on the reviewed training-split object-removal trajectories")
    b.add_argument("--inpainting", action="append", required=True, metavar="LABEL=HDF5", help="dump on the test split")
    for p in (a, b):
        p.add_argument("--main_ensembles", nargs=2, required=True, help="SayNo, then SayNo-CR (printed labels)")
        p.add_argument("--out", required=True, type=Path)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    {"detectors": detectors, "seen_unseen": seen_unseen}[args.figure](args)
    print(args.out)


if __name__ == "__main__":
    main()
