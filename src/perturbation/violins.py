"""Score distributions of the Bridge/Fractal detectors as violins, on one scale for every detector.

Scale: the percentile of a score among the ORIGINAL (unperturbed) inputs' scores of the same
detector, ties counted half. Originals are then uniform on [0, 1]; for perturbed inputs the mean
percentile is the Mann-Whitney AUROC of the thesis tables (tests/test_perturbation_score_violins.py),
and the share above 0.9 is the recall at the threshold that flags 10 % of the originals.

Two figures (THESIS profile, \\textwidth), both at the average of the initial and final frame:
  detector_figure     every row of tab:bf-detector-comparison x the three perturbations
  seen_unseen_figure  SayNo variants on Bridge training / test-seen-task / test-unseen-task demonstrations,
                      percentiles among the pooled test originals (one reference for all groups)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from say_no.perturbation.evaluate import paired_scores
from say_no.perturbation.instructions import seen_mask
from say_no.perturbation.load import ModelDump, load_detector_scores, load_models, restrict
from say_no.perturbation.names import MAIN_CLASSIFIERS, condition_name, mpl
from say_no.perturbation.novelty import encoder_novelty
from say_no.perturbation.scores import crossfit_folds
from say_no.perturbation.thesis_tables import classifier_group, clip_labels, detector_blocks
from say_no.utils.figstyle import FULL, THESIS, save_figure, style
from say_no.utils.palette import COLORS

FRAME = "both"
#: (condition, result set) of tab:bf-detector-comparison
DETECTOR_CONDITIONS = (("lang_swap_train", "val200"), ("lang_swap_near", "val200"), ("inpaint", "inpainting258"))
IMAGE_ONLY = ("clip_knn",)  # scores the image alone: undefined under a changed instruction
#: palette entity of the rows that are not a model of --color
ENTITY_COLORS = {"enc_knn": "encoder_novelty", "clip_alignment": "clip_alignment", "clip_knn": "clip_novelty"}
GROUP_TITLES = {classifier_group(False): "Classifiers"}  # the figure is narrower than the table
#: seen/unseen figure: groups as instruction_overlap.py (scope Bridge); it shows one relabeling variant only
GROUPS = (("train", "Training"), ("eval_seen", "Test,\nseen tasks"), ("eval_unseen", "Test,\nunseen tasks"))
REFERENCE_GROUPS = ("eval_seen", "eval_unseen")
SEEN_UNSEEN_CONDITIONS = ("lang_swap_train", "inpaint")
SCORE_SYMBOLS = {"spread": "$\\sigma_Q$", "neg_mean_q": "$-\\bar Q$"}
THRESHOLD = 0.9
HATCH = "/////"
GRID = np.linspace(0.0, 1.0, 401)
MIN_BANDWIDTH = 0.015
DETECTOR_HEIGHT, SEEN_UNSEEN_HEIGHT = 4.6, 4.4  # inches, at FULL width


def percentile_among(orig: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Share of `orig` below each x, ties counted half; the mean over perturbed x is the AUROC."""
    o = np.sort(np.asarray(orig, np.float64))
    if o.ndim != 1 or o.size == 0:
        raise ValueError(f"originals must be a non-empty 1-D array, got {o.shape}")
    lo, hi = np.searchsorted(o, x, "left"), np.searchsorted(o, x, "right")
    return (lo + 0.5 * (hi - lo)) / o.size


def bounded_kde(u: np.ndarray) -> np.ndarray:
    """Gaussian density of `u` on GRID, reflected at 0 and 1 (no mass outside [0, 1]), max = 1."""
    if u.min() < 0 or u.max() > 1:
        raise ValueError("bounded_kde needs values in [0, 1]")
    bw = max(0.25 * u.std(ddof=1), MIN_BANDWIDTH)
    k = lambda c: np.exp(-0.5 * ((GRID[:, None] - c[None, :]) / bw) ** 2).sum(1)
    d = k(u) + k(-u) + k(2 - u)
    return d / d.max()


def violin(ax, u: np.ndarray, pos: float, color: str, *, horizontal: bool, width: float, hatch: str | None) -> None:
    """Violin of percentiles `u` at `pos`, with its 10th / 50th / 90th percentile as bars across it."""
    half = width / 2 * bounded_kde(u)
    across, along = np.r_[pos + half, (pos - half)[::-1]], np.r_[GRID, GRID[::-1]]
    xs, ys = (along, across) if horizontal else (across, along)
    ax.fill(xs, ys, fc=color, ec=COLORS["neutral_dark"], lw=0.5, hatch=hatch, zorder=2)
    a, b = pos - 0.3 * width, pos + 0.3 * width
    for q, lw in zip(np.percentile(u, (10, 50, 90)), (0.7, 1.6, 0.7)):
        seg = ([q, q], [a, b]) if horizontal else ([a, b], [q, q])
        ax.plot(*seg, color=COLORS["aggregate"], lw=lw, solid_capstyle="butt", zorder=3)


def unit_axis(ax, axis: str) -> None:
    """[0, 1] percentile axis with the 10 %-of-originals threshold (dashed) and chance (dotted)."""
    getattr(ax, f"set_{axis}lim")(-0.03, 1.03)
    getattr(ax, f"set_{axis}ticks")([0, 0.5, 1])
    getattr(ax, f"{axis}axis").set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    line = ax.axvline if axis == "x" else ax.axhline
    line(THRESHOLD, color=COLORS["threshold"], ls="--", lw=0.8, zorder=1)
    line(0.5, color=COLORS["chance"], ls=":", lw=0.8, zorder=1)
    ax.grid(False)


def legend_handles() -> list:
    return [Line2D([], [], color=COLORS["aggregate"], lw=1.6, label="Median"),
            Line2D([], [], color=COLORS["aggregate"], lw=0.7, label="10th / 90th percentile"),
            Line2D([], [], color=COLORS["threshold"], ls="--", lw=0.8, label="0.9: flags 10 % of originals"),
            Line2D([], [], color=COLORS["chance"], ls=":", lw=0.8, label="0.5: chance")]


# --- data ---------------------------------------------------------------------------------------------

def load_detector_sets(sets: dict[str, list[tuple[str, Path]]], train_ref: dict[str, Path],
                       clip_scores: dict[str, Path], clip_label: str, ens: list[str], k: int) -> dict[str, dict]:
    """{set: {label: ModelDump}} with the ensembles' encoder novelty and the CLIP detectors, as
    analyse_perturbation_thesis.py builds them; all models of a set must describe the same rows."""
    if set(clip_scores) != set(sets):
        raise ValueError(f"CLIP scores for {sorted(clip_scores)}, sets {sorted(sets)}")
    out = {}
    for s, items in sets.items():
        ms = load_models(items)
        for lab in ens:
            ms.update(encoder_novelty(ms[lab], train_ref[lab], k))
        ms.update(load_detector_scores(clip_scores[s], {rd: lab for rd, lab in clip_labels(clip_label).items()}))
        ref = ms[ens[0]]
        for m in ms.values():
            if not (np.array_equal(m.trajectory_ids, ref.trajectory_ids) and np.array_equal(m.dataset, ref.dataset)):
                raise ValueError(f"{s}: {m.label} and {ref.label} describe different trajectories")
        out[s] = ms
    return out


def pair(m: ModelDump, cond: str, readout: str) -> tuple[np.ndarray, np.ndarray]:
    """(originals, perturbed) scores of one readout at FRAME; folds only feed readouts not used here."""
    return paired_scores(m, cond, FRAME, crossfit_folds(len(m.trajectory_ids), 2, 0))[readout]


def detector_rows(ens: list[str], clip_label: str, colors: dict[str, str]) -> list[dict]:
    """Rows of tab:bf-detector-comparison in its order: group, name, label, readout, colour, hatched."""
    clip = {**clip_labels(clip_label), "label": clip_label}
    blocks = detector_blocks(ens, list(MAIN_CLASSIFIERS), clip, encoder=True, clip_scores=("clip_alignment", "clip_knn"),
                             sole_relabeling=False, encoder_scores=("enc_knn",))
    rows = []
    for group, entries in blocks:
        for lab, rd, name in entries:
            model = lab.split(" | ")[0]
            key = colors[lab] if lab in colors else ENTITY_COLORS[rd]
            rows.append({"group": GROUP_TITLES.get(group, group), "name": name, "label": lab, "readout": rd,
                         "color": COLORS[key], "hatched": model == ens[1]})
    return rows


def detector_percentiles(sets: dict[str, dict], rows: list[dict]) -> dict[tuple[str, str, str], np.ndarray]:
    """{(model label, readout, condition): percentiles of the perturbed inputs}; image-only rows under
    object removal only."""
    out = {}
    for r in rows:
        for c, s in DETECTOR_CONDITIONS:
            if r["readout"] in IMAGE_ONLY and c != "inpaint":
                continue
            key = (r["label"], r["readout"], c)
            if key in out:
                raise ValueError(f"two rows draw {key}")
            out[key] = percentile_among(*pair(sets[s][r["label"]], c, r["readout"]))
    return out


def domain_groups(train: ModelDump, held_out: ModelDump, walk: dict) -> dict[str, ModelDump]:
    """Bridge rows: training-split demonstrations, test demonstrations whose instruction does / does not
    occur in training (instruction_overlap.py, scope 'bridge')."""
    if not seen_mask(train.language, train.dataset, walk, "exact").all():
        raise ValueError("a training-split trajectory has an instruction outside the training split")
    seen = seen_mask(held_out.language, held_out.dataset, walk, "exact")
    bridge = held_out.dataset == "bridge"
    return {"train": restrict(train, train.dataset == "bridge"), "eval_seen": restrict(held_out, bridge & seen),
            "eval_unseen": restrict(held_out, bridge & ~seen)}


def seen_unseen_percentiles(groups: dict[str, ModelDump], readout: str) -> dict[tuple[str, str, str], np.ndarray]:
    """{(group, 'orig' | condition, condition): percentiles among the pooled originals of REFERENCE_GROUPS}."""
    ref = np.concatenate([pair(groups[g], "inpaint", readout)[0] for g in REFERENCE_GROUPS])
    out = {}
    for g, m in groups.items():
        for c in SEEN_UNSEEN_CONDITIONS:
            o, p = pair(m, c, readout)
            out[(g, "orig", c)], out[(g, c, c)] = percentile_among(ref, o), percentile_among(ref, p)
    return out


# --- figures ------------------------------------------------------------------------------------------

def _row_positions(rows: list[dict]) -> tuple[list[float], list[tuple[str, float]]]:
    """y of every row (top first) and of every group header; a header takes 0.75 of a row."""
    y, ys, headers = 0.0, [], []
    for r in rows:
        if not headers or headers[-1][0] != r["group"]:
            y -= 0.25 if headers else 0.0
            headers.append((r["group"], y))
            y -= 0.75
        ys.append(y)
        y -= 1.0
    return ys, headers


def detector_figure(rows: list[dict], u: dict, out: Path, provenance: str) -> None:
    ys, headers = _row_positions(rows)
    with plt.rc_context(style(THESIS, width=FULL, height=DETECTOR_HEIGHT)):
        fig, axes = plt.subplots(1, len(DETECTOR_CONDITIONS), sharey=True, sharex=True)
        for ax, (c, _) in zip(axes, DETECTOR_CONDITIONS):
            for r, y in zip(rows, ys):
                if (r["label"], r["readout"], c) in u:
                    violin(ax, u[(r["label"], r["readout"], c)], y, r["color"], horizontal=True, width=0.82,
                           hatch=HATCH if r["hatched"] else None)
                else:
                    ax.text(0.5, y, "undefined", ha="center", va="center", fontstyle="italic",
                            color=COLORS["neutral_marker"], fontsize=plt.rcParams["xtick.labelsize"])
            ax.set_title(mpl(condition_name(c)).replace(" relabeling", "\nrelabeling").replace(" removal", "\nremoval"))
            unit_axis(ax, "x")
            ax.set_ylim(ys[-1] - 0.6, headers[0][1] + 0.35)
            for _, hy in headers[1:]:
                ax.axhline(hy + 0.5, color=COLORS["separator"], lw=0.6, zorder=0)
        axes[0].set_yticks(ys)
        axes[0].set_yticklabels([mpl(r["name"]) for r in rows])
        pad = plt.rcParams["ytick.major.pad"] + plt.rcParams["ytick.major.size"]
        for h, hy in headers:  # group headers: italic, no tick, right-aligned with the row names
            axes[0].annotate(mpl(h), (0, hy), xycoords=("axes fraction", "data"), xytext=(-pad, 0),
                             textcoords="offset points", ha="right", va="center", fontstyle="italic",
                             fontsize=plt.rcParams["ytick.labelsize"], annotation_clip=False)
        axes[1].set_xlabel("Percentile of perturbed input among original inputs")
        fig.legend(handles=legend_handles(), loc="outside lower center", ncol=4, frameon=False, handlelength=1.4,
                   columnspacing=1.0)
        save_figure(fig, out, provenance, preview=True)
        plt.close(fig)


def seen_unseen_figure(u: dict[tuple[str, str], dict], ens: list[str], out: Path, provenance: str) -> None:
    """u[(model label, readout)] = seen_unseen_percentiles(...); rows = model x readout."""
    rows = [(lab, rd) for rd in SCORE_SYMBOLS for lab in ens]
    with plt.rc_context(style(THESIS, width=FULL, height=SEEN_UNSEEN_HEIGHT)):
        fig, axes = plt.subplots(len(rows), len(SEEN_UNSEEN_CONDITIONS), sharex=True, sharey=True)
        for i, (lab, rd) in enumerate(rows):
            for j, c in enumerate(SEEN_UNSEEN_CONDITIONS):
                ax = axes[i, j]
                for g, (gk, _) in enumerate(GROUPS):
                    violin(ax, u[(lab, rd)][(gk, "orig", c)], g - 0.2, COLORS["original"], horizontal=False,
                           width=0.38, hatch=None)
                    violin(ax, u[(lab, rd)][(gk, c, c)], g + 0.2, COLORS["perturbed"], horizontal=False,
                           width=0.38, hatch=HATCH)
                unit_axis(ax, "y")
                if i == 0:
                    ax.set_title(mpl(condition_name(c, sole_relabeling=True)))
                if j == len(SEEN_UNSEEN_CONDITIONS) - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(f"{lab}, {SCORE_SYMBOLS[rd]}", rotation=270, va="bottom",
                                  fontsize=plt.rcParams["xtick.labelsize"])
        axes[-1, 0].set_xticks(range(len(GROUPS)))
        axes[-1, 0].set_xticklabels([name for _, name in GROUPS])
        fig.supylabel("Percentile among original test inputs", fontsize=plt.rcParams["axes.labelsize"])
        handles = [Patch(fc=COLORS["original"], ec=COLORS["neutral_dark"], lw=0.5, label="Original"),
                   Patch(fc=COLORS["perturbed"], ec=COLORS["neutral_dark"], lw=0.5, hatch=HATCH, label="Perturbed")]
        fig.legend(handles=handles + legend_handles(), loc="outside lower center", ncol=3, frameon=False,
                   handlelength=1.4, columnspacing=1.0)
        save_figure(fig, out, provenance, preview=True)
        plt.close(fig)
