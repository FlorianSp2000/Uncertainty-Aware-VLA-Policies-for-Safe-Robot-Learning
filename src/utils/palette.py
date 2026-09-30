"""One semantic colour per entity for every thesis figure: the same colour always means the same
model, score or input condition. Hues from Okabe & Ito (2008) and Paul Tol's muted / vibrant
schemes, chosen colour-blind-safe; greys for unperturbed inputs, black for aggregates.

Key -> entity (names as in the thesis):
  sayno_no_relabeling          SayNo (rho = 0)                               Okabe-Ito blue
  sayno_relabeling_rho10       SayNo-CR (rho = 10 %)                         Okabe-Ito sky blue
  sayno_relabeling_rho5        SayNo-CR (rho = 5 %)                          Tol indigo
  resnet_muse_classifier       ResNet-MUSE classifier (every variant)        Okabe-Ito bluish green
  octo_classifier              Octo classifier                               Okabe-Ito orange
  clip_novelty                 CLIP novelty (Mahalanobis or kNN)             Tol purple (CLIP family hue)
  clip_alignment               CLIP alignment (image-instruction cosine)    Tol purple (CLIP family hue; never in one
                               figure with CLIP novelty; not orange/pink, which mean Octo / perturbed)
  encoder_novelty              encoder novelty (SayNo's own features)        Okabe-Ito reddish purple
  ensemble_disagreement        score sigma_Q, when series are scores         Tol wine
  value_score                  score -Qbar, when series are scores           Tol olive
  time_normalised_value_score  score z, when series are scores               Tol sand
  original                     unperturbed input                             light grey
  true_premise                 IVA true-premise frame (= original)           light grey
  perturbed                    perturbed input (any perturbation)            Tol magenta
  false_premise_injection      IVA false-premise frame (= perturbed)         Tol magenta
  cross_task_relabeling        IVA counterfactual relabeling (other task)    Tol teal
  expected_execute             input whose expected decision is execute      Tol green (overview figure)
  aggregate                    average / pooled series of the components     black

Neutral greys for marks that are not an entity (the key names the role, not the grey):
  threshold                    a fitted detection threshold                  blue-grey
  chance                       chance / reference line (AUROC 0.5, profile end) mid grey
  bar_baseline                 the axis bars grow from (AUROC 0.5)           dark grey
  error_bar                    confidence-interval whiskers                  near black
  separator                    rule between row groups                       light grey
  grid                         gridlines of a quantitative axis              very light grey
  rug_background               empty cells of a label rug                    off-white
  neutral_dark                 a series with no palette entity               dark grey
  neutral_marker               legend key standing for any colour            grey
  stamp                        provenance footer                             grey

A sweep inside one entity (e.g. relabeling penalty) gets `sequential(key, n)`: same hue, lighter
to darker. Figure configs name colours through `resolve`.
"""
from __future__ import annotations

COLORS = {
    "sayno_no_relabeling": "#0072B2",
    "sayno_relabeling_rho10": "#56B4E9",
    "sayno_relabeling_rho5": "#332288",
    "resnet_muse_classifier": "#009E73",
    "octo_classifier": "#E69F00",
    "clip_novelty": "#AA4499",
    "clip_alignment": "#AA4499",
    "encoder_novelty": "#CC79A7",
    "ensemble_disagreement": "#882255",
    "value_score": "#999933",
    "time_normalised_value_score": "#DDCC77",
    "original": "#BBBBBB",
    "true_premise": "#BBBBBB",
    "perturbed": "#EE3377",
    "false_premise_injection": "#EE3377",
    "cross_task_relabeling": "#44AA99",
    "expected_execute": "#228833",
    "aggregate": "#000000",
    "threshold": "#7d8894",
    "chance": "#8c8c8c",
    "bar_baseline": "#4c4c4c",
    "error_bar": "#262626",
    "separator": "#cccccc",
    "grid": "#e3e3e3",
    "rug_background": "#f4f5f7",
    "neutral_dark": "#333333",
    "neutral_marker": "#595959",
    "stamp": "#737373",
}


def resolve(color: str | dict) -> str:
    """Colour spec of a figure config -> hex. Accepted: a palette key; '#rrggbb'; or
    {shade_of: <key>, index: i, n: n}, shade i of `sequential(key, n)`. Anything else raises,
    so a typo cannot fall back to matplotlib's default cycle."""
    if isinstance(color, dict):
        if set(color) != {"shade_of", "index", "n"}:
            raise ValueError(f"shade spec needs exactly shade_of, index, n; got {color}")
        shades = sequential(color["shade_of"], color["n"])
        if not 0 <= color["index"] < len(shades):
            raise ValueError(f"shade index {color['index']} outside 0..{len(shades) - 1}")
        return shades[color["index"]]
    if color in COLORS:
        return COLORS[color]
    if isinstance(color, str) and len(color) == 7 and color.startswith("#"):
        int(color[1:], 16)  # raises on a non-hex string
        return color
    raise ValueError(f"{color!r} is neither a palette key ({sorted(COLORS)}) nor '#rrggbb'")


#: share of white mixed into the lightest shade of `sequential`; lighter tints vanish on paper.
LIGHTEST_TINT = 0.6


def sequential(key: str, n: int) -> list[str]:
    """n ordered shades of one entity's hue, lightest first, the last equal to COLORS[key].

    For a sweep inside one entity (e.g. the relabeling penalty -1 / -10 / -100 of SayNo at
    rho = 5 %): hue says which entity, lightness says where in the sweep. Shades are tints
    (mixed with white), so the darkest is the entity's own colour and never drifts into a
    neighbouring entity's hue."""
    if key not in COLORS:
        raise ValueError(f"unknown palette key {key!r}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rgb = [int(COLORS[key][i:i + 2], 16) for i in (1, 3, 5)]
    shades = []
    for k in range(n):
        w = LIGHTEST_TINT * (1 - k / (n - 1)) if n > 1 else 0.0
        shades.append("#" + "".join(f"{round(c + (255 - c) * w):02X}" for c in rgb))
    return shades
