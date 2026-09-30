"""Reader-facing names of perturbations, detection scores and frames for the Bridge/Fractal thesis
tables and figures. Strings are LaTeX; `mpl` turns
them into matplotlib mathtext."""
from __future__ import annotations

import re

CONDITION_NAMES = {
    "lang_swap_train": "Random relabeling",
    "lang_swap_pkl": "Random relabeling (test pool)",
    "lang_swap_near": "Nearest-instruction relabeling",
    "inpaint": "Object removal",
    "jpeg_orig": "Re-encoding control",
    "act_far": "Far action",
    "act_other": "Other trajectory's action",
    "act_zero": "Zero action",
    "act_neg": "Negated action",
    "act_gripper": "Flipped gripper",
    "act_scaled": "Scaled action ($\\times 3$)",
    "act_uniform": "Uniform random action",
}
#: name of lang_swap_train in a float that shows no other relabeling variant; "Random relabeling" only
#: where it stands next to nearest-instruction relabeling (thesis terminology)
SOLE_RELABELING_NAME = "Counterfactual relabeling"
#: k of every kNN novelty score the names print (analyse_perturbation_thesis.py asserts --k equals it)
KNN_K = 5
#: model labels stored in perturbation_thesis.json by its analysis run -> the name the thesis prints
#: (the thesis, Method chapter: SayNo = rho = 0, SayNo-CR = counterfactual relabeling). The penalty
#: ablations are 4-member Bridge-only runs at rho = 5 %. Labels not listed print as stored.
MODEL_NAMES = {
    "No relabeling": "SayNo",
    "Penalty $-1$": "SayNo-CR, penalty $-1$",
    "Penalty $-1$ (repeat run)": "SayNo-CR, penalty $-1$ (repeat)",  # short: the table is at \textwidth
    "Penalty $-10$": "SayNo-CR, penalty $-10$",
    "Penalty $-100$": "SayNo-CR, penalty $-100$",
    "ResNet--MUSE 3 (per sample)": "ResNet--MUSE classifier",
    "Octo (per sample)": "Octo classifier",
}
#: classifiers every detector table and figure shows: the best of the three ResNet--MUSE training runs
#: (hyperparameter choice; all runs side by side in bf_classifier_variants_auroc) and the Octo classifier
MAIN_CLASSIFIERS = ("ResNet--MUSE 3 (per sample)", "Octo (per sample)")
SUPERSEDED_MODEL_NAMES = ("SayNo (no relabeling)", "SayNo (relabeling")
COMPOSITE = re.compile(r"^composite_(local|global)_r(\d+)$")
SCORE_NAMES = {
    "spread": "Ensemble disagreement $\\sigma_Q$",
    "neg_mean_q": "Value score $-\\bar Q$",
    # z~_p = (u_p - mu~_p) / s~_p with u_p = -Qbar_p at frame position p: a symbol of its own, since the z_t of
    # the conformal detector (the thesis, Method chapter) standardises against another profile; z~ = -z_old
    "neg_z": "Time-normalized, $\\tilde z$",
    "abs_z": "Time-normalized, $|\\tilde z|$",
    "z_gap": "Time-normalized, $\\tilde z_\\text{final} - \\tilde z_\\text{init}$",
    "enc_mahalanobis": "Mahalanobis distance",
    "enc_knn": "$k$-NN distance",
    "clip_mahalanobis": "CLIP novelty, Mahalanobis",
    "clip_knn": "CLIP novelty, $k$-NN",
    "clip_alignment": "CLIP alignment",
}
SCORE_SYMBOLS = {"spread": "$\\sigma_Q$", "neg_mean_q": "$-\\bar Q$"}
FRAME_NAMES = {"first": "Initial frame", "last": "Final frame", "both": "Average of both frames"}
SMALL_WORDS = {"of", "and", "or", "per", "the", "to", "at", "in", "vs."}


def model_name(label: str) -> str:
    """Printed name of a stored model label; a superseded label fails instead of reaching a float."""
    if label.startswith(SUPERSEDED_MODEL_NAMES):
        raise ValueError(f"superseded model label {label!r}; see MODEL_NAMES")
    return MODEL_NAMES.get(label, label)


def condition_name(c: str, sole_relabeling: bool = False) -> str:
    """`sole_relabeling`: the float shows lang_swap_train as its only relabeling variant."""
    if sole_relabeling and c == "lang_swap_train":
        return SOLE_RELABELING_NAME
    if sole_relabeling and c in ("lang_swap_pkl", "lang_swap_near"):
        raise ValueError(f"{c} is a relabeling variant of its own; the float is not single-variant")
    m = COMPOSITE.match(c)
    if m:
        return "Edit-region control" if m.group(1) == "local" else "Background control"
    if c not in CONDITION_NAMES:
        raise KeyError(f"no reader-facing name for condition {c!r}")
    return CONDITION_NAMES[c]


def title_case(s: str) -> str:
    """Column-header capitalisation; math ($...$) and parenthesised symbols stay as they are."""
    parts = re.split(r"(\$[^$]*\$)", s)
    out = []
    for p in parts:
        if p.startswith("$"):
            out.append(p)
            continue
        words = re.split(r"([ \-])", p)
        out.append("".join(w if w.lower().strip("()") in SMALL_WORDS and i > 0 else _cap(w)
                           for i, w in enumerate(words)))
    return "".join(out)


def _cap(w: str) -> str:
    i = next((k for k, ch in enumerate(w) if ch.isalpha()), None)
    return w if i is None else w[:i] + w[i].upper() + w[i + 1:]


def mpl(s: str) -> str:
    """LaTeX label -> matplotlib text (mathtext keeps $...$; '--' is the en dash)."""
    return s.replace("--", "–")
