"""Figure style and saving for the two targets: the thesis and the ICRA paper.

A profile fixes what a target decides -- reference widths, font scale over the tueplots icml2024
bundle, legend size, font family, usetex. Drawing code takes a profile from its caller and never
carries its own width or font constants. Widths are fractions of the profile's \\columnwidth
(thesis: one column, so \\textwidth), equal to the `width=` the figure is included at, so LaTeX
does not rescale it and the rc sizes are the sizes on the page.

Saving: `save_figure` for the thesis -- exact size (no tight bbox), provenance (+ render date) in the
PDF metadata, never on the page (run ids and paths mean nothing to a thesis reader), no
CreationDate, optional preview PNG. `stamp` draws a visible footer for outputs outside the thesis. `save_pinned` for outputs whose bytes are pinned
(paper profile, legacy W&B YAMLs): exactly `fig.savefig`, so they stay byte-identical.
Registry: src/scripts/render_thesis_figures.py.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.text import Text

from tueplots import bundles

from say_no.utils.palette import COLORS

#: \textwidth of thesis/msc_thesis.tex (scrbook, a4paper, BCOR1cm; from msc_thesis.log)
THESIS_TEXTWIDTH_PT = 398.33864
FULL, HALF = 1.0, 0.49
STAMP_PT = 4.5
STAMP_PAD_IN = 0.02
PREVIEW_DPI = 200
TIMES = ("Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman")
#: the thesis body font (URW Palladio); figures match it, maths included
PALATINO = ("Palatino Linotype", "TeX Gyre Pagella", "URW Palladio L", "Palatino")


#: rc keys a tueplots bundle sets in points; scaled together so their ratios survive.
FONT_SIZE_KEYS = (
    "font.size", "axes.labelsize", "axes.titlesize", "figure.titlesize",
    "legend.fontsize", "legend.title_fontsize", "xtick.labelsize", "ytick.labelsize",
)


def get_style(bundle: str = "icml2024", *, font_scale: float = 1.0, **bundle_kwargs) -> dict:
    """Return a matplotlib rc dict from a tueplots bundle (icml2024, icml2022, neurips2024, ...).

    Pass into `plt.rc_context(...)` or `plt.rcParams.update(...)`.

    `font_scale` multiplies every font size in the bundle. A panel drawn at a figsize
    smaller than the width it is included at gets magnified by LaTeX, which shrinks its
    text on the page; scaling the fonts up here cancels that.
    """
    if font_scale <= 0:
        raise ValueError(f"font_scale must be positive, got {font_scale}")
    fn = getattr(bundles, bundle, None)
    if fn is None:
        raise ValueError(f"Unknown tueplots bundle: {bundle!r}")
    rc = fn(**bundle_kwargs)
    for key in FONT_SIZE_KEYS:
        if key in rc:
            rc[key] = rc[key] * font_scale
    return rc


@dataclass(frozen=True)
class Profile:
    name: str
    column_in: float          # \columnwidth, the reference of `width`
    text_in: float            # \textwidth (span="text", e.g. a figure* in two-column ICRA)
    font_scale: float         # over the icml2024 bundle's point sizes
    legend_pt: float | None   # None: the bundle's legend size x font_scale
    serif: tuple[str, ...]    # font family, first available wins
    usetex: bool
    serif_math: bool = False  # mathtext in the text font (else the bundle's Computer Modern)
    pdf_fonttype: int | None = None  # 42 embeds TrueType; None leaves the rc (pinned ICRA bytes)


THESIS = Profile("thesis", THESIS_TEXTWIDTH_PT / 72.27, THESIS_TEXTWIDTH_PT / 72.27, 1.3, None, PALATINO, False,
                 serif_math=True, pdf_fonttype=42)
ICRA = Profile("icra", 3.4, 7.0, 1.3, 9.0, TIMES, False)
PROFILES = {p.name: p for p in (THESIS, ICRA)}


def style(profile: Profile, *, width: float, height: float, span: str = "column",
          usetex: bool | None = None) -> dict:
    """rc dict for plt.rc_context: the profile's fonts and a figure of `width` x profile's
    column (or text) width by `height` inches."""
    if span not in ("column", "text"):
        raise ValueError(f"span must be column or text, got {span!r}")
    if not 0 < width <= (profile.text_in / profile.column_in if span == "column" else 1.0):
        raise ValueError(f"width {width} is not a fraction of the {profile.name} {span} width")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    rc = get_style("icml2024", column="half", usetex=profile.usetex if usetex is None else usetex,
                   font_scale=profile.font_scale)
    rc["font.serif"] = list(profile.serif)
    if profile.pdf_fonttype is not None:
        rc["pdf.fonttype"] = profile.pdf_fonttype
    if profile.serif_math:
        name = installed_serif(profile)
        rc.update({"mathtext.fontset": "custom", "mathtext.rm": name, "mathtext.it": f"{name}:italic",
                   "mathtext.bf": f"{name}:bold", "mathtext.sf": name})
    ref = profile.column_in if span == "column" else profile.text_in
    rc["figure.figsize"] = (width * ref, height)
    return rc


def installed_serif(profile: Profile) -> str:
    """First font of the profile's family that is installed; raises instead of a silent fallback."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in profile.serif:
        if name in installed:
            return name
    raise RuntimeError(f"none of {profile.serif} installed (profile {profile.name})")


def legend_kwargs(profile: Profile) -> dict:
    """fontsize for a legend: the profile's own size, or nothing (the rc size)."""
    return {} if profile.legend_pt is None else {"fontsize": profile.legend_pt}


def value_grid(ax, axis: str) -> None:
    """Light major gridlines along the value axis ("x", "y" or "both") of a quantitative plot, behind
    the data; example strips and image panels get none."""
    if axis not in ("x", "y", "both"):
        raise ValueError(f"axis must be x, y or both, got {axis!r}")
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, which="major", color=COLORS["grid"], lw=0.5)


def thesis_text_sizes() -> set[float]:
    """Every point size a thesis figure may print: labels/titles, ticks/legend."""
    rc = style(THESIS, width=FULL, height=1.0)
    return {rc["axes.labelsize"], rc["xtick.labelsize"], rc["legend.fontsize"]}


def _render_date() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    d = dt.datetime.fromtimestamp(int(epoch), dt.timezone.utc).date() if epoch else dt.date.today()
    return d.isoformat()


def provenance_record(provenance: str) -> str:
    """`provenance · rendered <date>`: the thesis figure metadata and the visible stamp."""
    if not provenance or not provenance.strip():
        raise ValueError("a figure needs a provenance record")
    return f"{provenance} · rendered {_render_date()}"


def stamp(fig: Figure, provenance: str) -> Text:
    """Provenance footer, bottom left inside the canvas: `provenance · rendered <date>` at STAMP_PT.
    Raises if it does not fit the canvas width (it would be cut off)."""
    record = provenance_record(provenance)
    w_in, h_in = fig.get_size_inches()
    t = fig.text(STAMP_PAD_IN / w_in, STAMP_PAD_IN / h_in, record,
                 fontsize=STAMP_PT, color=COLORS["stamp"], ha="left", va="bottom", gid="stamp")
    fig.canvas.draw()
    bb, canvas = t.get_window_extent(), fig.bbox
    if bb.x1 > canvas.x1 or bb.y1 > canvas.y1:
        raise ValueError(f"stamp does not fit the {w_in:.2f} in canvas: {t.get_text()!r}")
    return t


def save_figure(fig: Figure, path: str | Path, provenance: str, *, preview: bool = False,
                dpi: float | None = None) -> None:
    """Thesis figure to `path` (PDF) at its exact size; `provenance` and the render date go into the
    PDF metadata (Subject), not onto the page. `preview` also writes path.with_suffix('.png') at
    PREVIEW_DPI (record in its Description). `dpi` is the raster resolution of embedded images
    (default: the rc's)."""
    record = provenance_record(provenance)
    if list(matplotlib.rcParams["font.serif"]) != list(THESIS.serif) or matplotlib.rcParams["text.usetex"] != THESIS.usetex:
        raise RuntimeError("save_figure must run inside plt.rc_context(style(THESIS, ...)), where the figure was drawn")
    path = Path(path)
    if path.suffix != ".pdf":
        raise ValueError(f"thesis figures are PDFs, got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {} if dpi is None else {"dpi": dpi}
    fig.savefig(path, metadata={"CreationDate": None, "Subject": record}, **kw)
    if preview:
        fig.savefig(path.with_suffix(".png"), dpi=PREVIEW_DPI, metadata={"Description": record})


def save_pinned(fig: Figure, path: str | Path, **savefig_kwargs) -> None:
    """Outputs outside the thesis contract -- paper-profile and legacy YAML figures, diagnostic
    sheets: plain savefig."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **savefig_kwargs)
