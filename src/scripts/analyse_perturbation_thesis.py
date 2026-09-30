"""Thesis tables and figures for Bridge/Fractal from the perturbation-sensitivity dumps: every model
(Q-ensembles, reward ablations, classifiers, Octo), the CLIP detectors and novelty in the ensembles'
own encoder, AUROC [paired CI] per condition x frame x readout (+ per source dataset), operating
points, member subsets, shift correlations. Library: say_no.perturbation. Invocation with the paths
used for the thesis: docs/REPRODUCE.md, step 2b.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from say_no.perturbation import figures
from say_no.perturbation import thesis_tables as T
from say_no.perturbation.evaluate import shift_spearman
from say_no.perturbation.load import load_detector_scores, load_models
from say_no.perturbation.names import KNN_K
from say_no.perturbation.novelty import encoder_novelty
from say_no.perturbation.report import analyse

SETS = ("val200", "inpainting258")


def labelled(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"expected LABEL=VALUE, got {it!r}")
        out.append(tuple(it.rsplit("=", 1)))  # labels may contain '=' (LaTeX), values may not
    return out


def paths(items: list[str]) -> list[tuple[str, Path]]:
    out = [(lab, Path(p)) for lab, p in labelled(items)]
    for _, p in out:
        if not p.exists():
            raise FileNotFoundError(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--val", action="append", required=True, metavar="LABEL=HDF5", help="dump on the language-swap set")
    ap.add_argument("--inpainting", action="append", required=True, metavar="LABEL=HDF5", help="dump on the inpainting set")
    ap.add_argument("--train_ref", action="append", required=True, metavar="LABEL=HDF5",
                    help="ensemble dump of training-split originals with feat/: calibration + encoder reference")
    ap.add_argument("--clip_scores", action="append", required=True, metavar="SET=HDF5")
    ap.add_argument("--clip_label", required=True)
    ap.add_argument("--main_ensembles", nargs="+", required=True,
                    help="labels of the two SayNo variants (SayNo first, then SayNo-CR); every LABEL is the reader-facing "
                         "LaTeX name that tables and figures print")
    ap.add_argument("--classifiers", nargs="+", required=True)
    ap.add_argument("--classifier_note", action="append", required=True, metavar="LABEL=TEXT")
    ap.add_argument("--color", action="append", required=True, metavar="LABEL=KEY",
                    help="say_no.utils.palette key of each SayNo variant and classifier")
    ap.add_argument("--ablations", nargs="+", required=True)
    ap.add_argument("--reference_ensemble", required=True)
    ap.add_argument("--latency", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--k", type=int, required=True, help="k of the encoder-feature kNN")
    ap.add_argument("--n_boot", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_folds", type=int, required=True, help="cross-fitting folds for the per-frame z")
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--table_alpha", required=True, help="alpha of the operating point in the inpainting table")
    ap.add_argument("--n_splits", type=int, required=True)
    ap.add_argument("--op_condition", required=True)
    ap.add_argument("--member_sizes", type=int, nargs="+", required=True)
    ap.add_argument("--n_subsets", type=int, required=True)
    ap.add_argument("--frame_readouts", nargs="+", required=True)
    ap.add_argument("--actions", nargs="+", required=True)
    args = ap.parse_args()
    if args.k != KNN_K:
        raise ValueError(f"--k {args.k}: the table and figure names print {KNN_K}-NN (say_no.perturbation.names.KNN_K)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = {"val200": load_models(paths(args.val)), "inpainting258": load_models(paths(args.inpainting))}
    train = load_models(paths(args.train_ref))
    clip = dict(paths(args.clip_scores))
    if set(clip) != set(SETS):
        raise ValueError(f"--clip_scores needs {SETS}, got {list(clip)}")
    for s in SETS:
        ms = models[s]
        for lab in args.main_ensembles:
            ms.update(encoder_novelty(ms[lab], [p for l, p in paths(args.train_ref) if l == lab][0], args.k))
        ms.update(load_detector_scores(clip[s], {rd: lab for rd, lab in T.clip_labels(args.clip_label).items()}))
        ref = ms[args.main_ensembles[0]]
        for m in ms.values():
            if list(m.trajectory_ids) != list(ref.trajectory_ids) or list(m.dataset) != list(ref.dataset):
                raise ValueError(f"{s}: {m.label} and {ref.label} describe different trajectories")
        if not all(list(m.swap_language.get(c, ref.swap_language[c])) == list(ref.swap_language[c])
                   for m in ms.values() for c in ref.swap_language):
            raise ValueError(f"{s}: models scored different swapped instructions")
    needed = set(args.main_ensembles) | set(args.classifiers) | set(args.ablations) | {args.reference_ensemble}
    missing = {(s, l) for s in SETS for l in needed if l not in models[s]}
    if missing:
        raise ValueError(f"labels without a dump: {sorted(missing)}")

    res = {s: analyse(models[s], {l: train[l] for l in args.main_ensembles}, n_boot=args.n_boot, seed=args.seed,
                      n_folds=args.n_folds, alphas=args.alphas, n_splits=args.n_splits, op_condition=args.op_condition,
                      member_sizes=args.member_sizes, n_subsets=args.n_subsets,
                      subset_conditions=["lang_swap_train", "inpaint"],
                      dataset_conditions=list(T.LANG) + ["inpaint"]) for s in SETS}
    spear = {s: {lab: {c: {rd: shift_spearman(models[s][f"{lab} | {rd}"], models[s][lab], c, args.n_boot, args.seed)
                           for rd in ("enc_mahalanobis", "enc_knn")}
                       for c in models[s][f"{lab} | enc_knn"].conditions if c != "orig"}
                 for lab in args.main_ensembles} for s in SETS}

    ens, cls = args.main_ensembles, args.classifiers
    st = models["inpainting258"][ens[0]].meta[0]["image_stats"]
    masks = {"mask_threshold": st["mask_threshold"], "area_median": st["mask_area_fraction_r16"]["median"],
             "jpeg_quality": st["jpeg"]["chosen_quality"], "median_delta": st["jpeg"]["target_median_abs_delta_inpaint"]}
    roles = {"main_ensembles": ens, "classifiers": cls, "classifier_notes": dict(labelled(args.classifier_note)),
             "ablations": args.ablations, "reference_ensemble": args.reference_ensemble, "clip_label": args.clip_label,
             "table_alpha": args.table_alpha, "actions": args.actions, "colors": dict(labelled(args.color))}
    if set(roles["colors"]) != set(ens) | set(cls):
        raise ValueError("--color needs exactly the SayNo variants and classifiers")
    o = args.out_dir
    lat = T.write_all(res, spear, masks, roles, args.latency, o)

    figures.write_all(res, roles, o, "Bridge/Fractal perturbation dumps (controls/, 2026-09-28) · "
                      "analyse_perturbation_thesis.py", args.frame_readouts)

    out = o / "perturbation_thesis.json"
    out.write_text(json.dumps({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                               "sets": res, "shift_spearman": spear, "latency": lat, "controls_caption": masks, "roles": roles},
                              indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
