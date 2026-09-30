"""Thesis figures of the Bridge/Fractal perturbation analysis, in the THESIS profile of
say_no.utils.figstyle at full \\textwidth (the width they are included at).

Colours: say_no.utils.palette (one colour per entity across the thesis); a model's key is its
`roles["colors"]` entry. Frames are told apart by marker, and the average of both frames is a black
(aggregate) diamond so it never reads as a third frame.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from say_no.perturbation.names import FRAME_NAMES, MAIN_CLASSIFIERS, SCORE_NAMES, SCORE_SYMBOLS, condition_name, model_name, mpl
from say_no.perturbation.thesis_tables import CONTROLS, MAIN_LANG, clip_labels
from say_no.utils.figstyle import FULL, THESIS, save_figure, style, value_grid
from say_no.utils.palette import COLORS

FRAME_MARKERS = {"first": dict(marker="o", mfc="white"), "last": dict(marker="s"),
                 "both": dict(marker="D", color=COLORS["aggregate"], mfc=COLORS["aggregate"])}
CHANCE = dict(color=COLORS["chance"], lw=0.6, ls=":")
NEUTRAL = COLORS["neutral_marker"]  # legend keys that stand for every model's colour
TICK_NAMES = {"spread": "Ensemble\ndisagreement\n$\\sigma_Q$", "neg_mean_q": "Value\nscore\n$-\\bar Q$"}


def _style(height: float, width: float = FULL) -> dict:
    return style(THESIS, width=width, height=height)


def _cell(r: dict, cond: str, frame: str, rd: str) -> dict:
    c = r["auroc"].get(cond, {}).get(frame, {}).get(rd)
    if c is None:
        raise KeyError(f"no AUROC for {cond}/{frame}/{rd}")
    return c


def _err(cells: list[dict]) -> np.ndarray:
    return np.array([[c["auroc"] - c["ci95"][0] for c in cells], [c["ci95"][1] - c["auroc"] for c in cells]])


def frame_split(results: dict, models: list[tuple[str, str]], conditions: list[str], scores: list[str],
                out: Path, provenance: str) -> None:
    """AUROC [95 % CI] of each score at the initial frame, the final frame and their average;
    rows = model (label, colour), columns = perturbation."""
    off = {"first": -0.22, "last": 0.0, "both": 0.22}  # centred on the tick
    with plt.rc_context(_style(1.45 * len(models) + 0.55)):
        fig, axes = plt.subplots(len(models), len(conditions), sharey=True, sharex=True, squeeze=False)
        x = np.arange(len(scores))
        for i, (lab, color) in enumerate(models):
            for j, c in enumerate(conditions):
                ax = axes[i][j]
                for f in ("first", "last", "both"):
                    cells = [_cell(results[lab], c, f, rd) for rd in scores]
                    style = {"color": color, "mec": color if f != "both" else COLORS["aggregate"], **FRAME_MARKERS[f]}
                    ax.errorbar(x + off[f], [e["auroc"] for e in cells], yerr=_err(cells), ls="none", ms=3.6,
                                mew=0.9, elinewidth=0.8, capsize=0, **style)
                ax.axhline(0.5, **CHANCE)
                value_grid(ax, "y")
                ax.set_ylim(0.25, 1.0)
                ax.set_xlim(-0.6, len(scores) - 0.4)
                ax.set_xticks(x)
                ax.set_xticklabels([TICK_NAMES[rd] for rd in scores])
                if i == 0:
                    ax.set_title(mpl(condition_name(c)).replace(" ", "\n", 1))
                if j == 0:
                    ax.set_ylabel(mpl(model_name(lab)).replace(" (", "\n(") + "\nAUROC")
        handles = [Line2D([], [], ls="none", ms=4, mew=0.9, color=NEUTRAL, mec=NEUTRAL, **FRAME_MARKERS["first"]),
                   Line2D([], [], ls="none", ms=4, mew=0.9, color=NEUTRAL, mec=NEUTRAL, **FRAME_MARKERS["last"]),
                   Line2D([], [], ls="none", ms=4, mew=0.9, mec=COLORS["aggregate"], **FRAME_MARKERS["both"])]
        fig.legend(handles, [FRAME_NAMES[f] for f in ("first", "last", "both")], loc="outside upper center",
                   ncol=3, frameon=False)
        save_figure(fig, out, provenance)
        plt.close(fig)


#: marker per score family, so rows stay distinguishable in black-and-white print
SCORE_MARKERS = {"spread": "o", "neg_mean_q": "s", "enc_knn": "D", "p_infeasible": "^", "clip_alignment": "v",
                 "clip_knn": "P"}


def detector_dots(panels: list[tuple[str, dict, str]], groups: list[list[tuple[str, str, str, str]]],
                  out: Path, provenance: str, undefined: tuple[tuple[str, str], ...] = ()) -> None:
    """Small multiples: one panel per perturbation (title, result set, condition), one row per detector,
    AUROC [95 % CI] as a dot on a shared axis from 0.4 (lower only if a CI reaches below) to 1.
    groups: [[(label, score, row name, colour)]], separated by a rule. `undefined`: (score, condition)
    pairs with no AUROC by construction, marked n/a."""
    n_rows = sum(len(rows) for rows in groups)
    with plt.rc_context(_style(0.2 * n_rows + 0.85)):
        cis = [_cell(rs[lab], c, "both", rd)["ci95"][0] for _, rs, c in panels for rows in groups
               for lab, rd, _, _ in rows if (rd, c) not in undefined]
        lo = min(0.4, np.floor(min(cis) * 5) / 5)
        ticks_x = [t for t in (0.25, 0.5, 0.75, 1.0) if t >= lo]  # chance labelled; no label at the panel edge lo
        fig, axes = plt.subplots(1, len(panels), sharey=True, layout="constrained", squeeze=False)
        for ax, (title, res_set, cond) in zip(axes[0], panels):
            y, ticks, labels = n_rows - 1, [], []
            for g, rows in enumerate(groups):
                if g:
                    ax.axhline(y + 0.5, color=COLORS["separator"], lw=0.5)
                for lab, rd, name, color in rows:
                    if (rd, cond) in undefined:
                        ax.text(0.7, y, "n/a", color=COLORS["neutral_marker"], ha="center", va="center",
                                fontsize=plt.rcParams["xtick.labelsize"])
                    else:
                        cell = _cell(res_set[lab], cond, "both", rd)
                        ax.errorbar([cell["auroc"]], [y], xerr=_err([cell]), ls="none", marker=SCORE_MARKERS[rd],
                                    ms=3.6, mew=0.8, color=color, elinewidth=0.9, capsize=1.6, capthick=0.9,
                                    zorder=3, clip_on=False)
                    ticks.append(y); labels.append(mpl(name))
                    y -= 1
            ax.axvline(0.5, **CHANCE)
            value_grid(ax, "x")
            ax.set_xlim(lo, 1.02)  # headroom: markers at AUROC ~1 stay off the spine
            ax.set_xticks(ticks_x)
            ax.set_xticklabels([f"{t:g}" for t in ticks_x])
            ax.set_ylim(-0.6, n_rows - 0.4)
            ax.set_title(mpl(title).replace(" ", "\n", 1))
            ax.set_xlabel("AUROC")
            ax.tick_params(axis="y", length=0)
        axes[0][0].set_yticks(ticks)
        axes[0][0].set_yticklabels(labels, va="center_baseline")
        save_figure(fig, out, provenance, preview=True)
        plt.close(fig)


FILES = {"frames": "auroc_initial_vs_final_frame.pdf", "detectors": "auroc_detectors_by_perturbation.pdf",
         "controls": "auroc_object_removal_vs_controls.pdf"}


def write_all(res: dict, roles: dict, out: Path, provenance: str, frame_scores: list[str]) -> None:
    """The three AUROC figures from the analysis dict (`perturbation_thesis.json` "sets" + "roles")."""
    ens, cls = roles["main_ensembles"], list(MAIN_CLASSIFIERS)
    if not set(cls) <= set(roles["classifiers"]):
        raise ValueError(f"MAIN_CLASSIFIERS {cls} not all in the analysis classifiers {roles['classifiers']}")
    if len(ens) != 2:
        raise ValueError("expected two SayNo variants: SayNo, SayNo-CR")
    colors = {lab: COLORS[key] for lab, key in roles["colors"].items()}
    clip = clip_labels(roles["clip_label"])
    val, inp = res["val200"], res["inpainting258"]

    merged = {lab: {"auroc": {**{c: val[lab]["auroc"][c] for c in MAIN_LANG}, "inpaint": inp[lab]["auroc"]["inpaint"]}}
              for lab in ens}
    frame_split(merged, [(lab, colors[lab]) for lab in ens], list(MAIN_LANG) + ["inpaint"], frame_scores,
                out / FILES["frames"], provenance)

    enc = COLORS["encoder_novelty"]
    sym = SCORE_SYMBOLS
    # flat row labels (model, score) instead of header rows: one alignment, no empty rows
    rows = [[(lab, rd, f"{model_name(lab)}, {sym[rd]}", colors[lab]) for rd in sym] for lab in ens]
    rows.append([(f"{lab} | enc_knn", "enc_knn", f"Encoder novelty ({model_name(lab)})", enc) for lab in ens])
    rows.append([(lab, "p_infeasible", model_name(lab), colors[lab]) for lab in cls])  # training in the caption
    clip_rows = [(clip["clip_alignment"], "clip_alignment", SCORE_NAMES["clip_alignment"], COLORS["clip_alignment"]),
                 (clip["clip_knn"], "clip_knn", "CLIP novelty", COLORS["clip_novelty"])]
    image_only = tuple(("clip_knn", c) for c in MAIN_LANG)  # scores the image alone: undefined under relabeling
    detector_dots([(condition_name(c), val, c) for c in MAIN_LANG] + [(condition_name("inpaint"), inp, "inpaint")],
                  rows + [clip_rows], out / FILES["detectors"], provenance, undefined=image_only)

    best_cls = max(cls, key=lambda l: inp[l]["auroc"]["inpaint"]["both"]["p_infeasible"]["auroc"])
    controls = [[(lab, "spread", f"{model_name(lab)}, {sym['spread']}", colors[lab]) for lab in ens] +
                 [(f"{ens[0]} | enc_knn", "enc_knn", "Encoder novelty", enc),
                  (best_cls, "p_infeasible", model_name(best_cls), colors[best_cls]), clip_rows[1]]]
    detector_dots([(condition_name(c), inp, c) for c in CONTROLS], controls, out / FILES["controls"], provenance)
